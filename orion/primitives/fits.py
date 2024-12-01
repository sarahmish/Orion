"""Primitive Anomaly Transformer

This primitive is an pytorch implementation of "FITS: Modeling 
Time Series with 10k parameters"
https://arxiv.org/pdf/2307.03756

This is a modified version of the original code, which can be found
at https://github.com/VEWOXIC/FITS
"""
# -*- coding: utf-8 -*-

import logging
import math
import operator
import os
from itertools import groupby
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from mlstars.utils import import_object
from torch.utils.data import DataLoader

LOGGER = logging.getLogger(__name__)

class Signal(object):
    """Data object.

    Args:
        X (ndarray):
            An n-dimensional array of signal values.
        window_size (int):
            Size of the window.
        step (int):
            Stride size.
    """

    def __init__(self, X, window_size, step=1, mode='train'):
        self.data = X
        self.step = step
        self.mode = mode
        self.window_size = window_size

    def __len__(self):
        return (self.data.shape[0] - self.window_size) // self.step + 1

    def __getitem__(self, index):
        start = index * self.step
        end = start + self.window_size

        if self.mode == 'train' or self.mode == 'test':
            return np.float32(self.data[start: end])
        else:
            raise ValueError(f'Unknown {self.mode} mode.')


class Model(nn.Module):
    # FITS: Frequency Interpolation Time Series Forecasting
    def __init__(self, seq_len, pred_len, individual, channels, cut_freq):
        super(Model, self).__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.individual = individual
        self.channels = channels

        # Decompsition Kernel Size
        kernel_size = 25
        self.dominance_freq = cut_freq # 720/24
        self.length_ratio = (self.seq_len + self.pred_len) / self.seq_len

        if self.individual:
            self.freq_upsampler = nn.ModuleList()
            for i in range(self.channels):
                self.freq_upsampler.append(nn.Linear(self.dominance_freq, int(self.dominance_freq*self.length_ratio)).to(torch.cfloat))

        else:
            self.freq_upsampler = nn.Linear(self.dominance_freq, int(self.dominance_freq*self.length_ratio)).to(torch.cfloat)


    def forward(self, x):
        # RIN
        x_mean = torch.mean(x, dim=1, keepdim=True)
        x = x - x_mean
        x_var = torch.var(x, dim=1, keepdim=True)+ 1e-5
        x = x / torch.sqrt(x_var)

        low_specx = torch.fft.rfft(x, dim=1)
        low_specx[:,self.dominance_freq:]=0
        low_specx = low_specx[:,0:self.dominance_freq,:]
        if self.individual:
            low_specxy_ = torch.zeros([low_specx.size(0),int(self.dominance_freq*self.length_ratio),low_specx.size(2)],dtype=low_specx.dtype).to(low_specx.device)
            for i in range(self.channels):
                low_specxy_[:,:,i]=self.freq_upsampler[i](low_specx[:,:,i].permute(0,1)).permute(0,1)
        else:
            low_specxy_ = self.freq_upsampler(low_specx.permute(0,2,1)).permute(0,2,1)

        low_specxy = torch.zeros([low_specxy_.size(0),int((self.seq_len+self.pred_len)/2+1),low_specxy_.size(2)],dtype=low_specxy_.dtype).to(low_specxy_.device)
        low_specxy[:,0:low_specxy_.size(1),:]=low_specxy_
        low_xy=torch.fft.irfft(low_specxy, dim=1)
        low_xy=low_xy * self.length_ratio
        
        xy = (low_xy) * torch.sqrt(x_var) +x_mean
        return xy, low_xy * torch.sqrt(x_var)
    
class FITS():
    """Anomaly Transformer model for unsupervised time series anomaly detection.

    Args:
        window_size (int):
            Window size of each sample.
        step (int):
            Stride size between samples.
        unit (str):
            String representing the unit of timestamps.
        interval (int):
            The time gap between one sample and another.
        input_size (int):
            Input size for the network.
        output_size (int):
            Output size for the network.
        d_model (int):
            Model dimension.
        n_hidden (int):
            Hidden dimension.
        batch_size (int):
            Number of example per epoch.
        dropout (float):
            Dropout value of the network.
        attention_dropout (float):
            Dropout value for attention.
        epochs (int):
            Number of iterations to train the model.
        learning_rate (float):
            Learning rate for the optimizer.
        temperature (int):
            Scaling value. Default 50.
        verbose (bool):
            Whether to be on verbose mode or not.
        cuda (bool):
            Whether to use GPU or not.
        valid_split (float):
            A float to split data dataframe to validation set. Data needs to contain a label
            column. Use ``target_column`` to change the target column name.
        output_dir (str):
            Path to folder where to save the model.
    """
    @staticmethod
    def _adjust_learning_rate(optimizer, epoch, lr_):
        lr_adjust = {epoch: lr_ * (0.5 ** ((epoch - 1) // 1))}
        if epoch in lr_adjust.keys():
            lr = lr_adjust[epoch]
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

            LOGGER.info(f'Updating learning rate to {lr}')

    def __init__(self, input_size=1, output_size=1, window_size=200, step=1, cut_freq=25, 
                 individual=False, DSR=4, batch_size=1024, learning_rate=1e-4, epochs=10, 
                 valid_split=0.2, shuffle=True, cuda=True, optimizer="torch.optim.Adam", 
                 verbose=False, output_dir=False):
        
        assert (window_size / DSR) / 2 >= cut_freq, 'cutfreq should be smaller than half of the window size after downsampling'
        if cut_freq == 0:
            cut_freq = int(window_size / DSR / 2)

        self.input_size = input_size
        self.output_size = output_size
        self.window_size = window_size
        self.DSR = DSR
        self.cut_freq = cut_freq
        self.individual = individual
        self.step = step

        self.seq_len = self.window_size // self.DSR
        self.pred_len = self.window_size - self.window_size // self.DSR

        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.valid_split = valid_split
        self.shuffle = shuffle
        self.cuda = cuda
        self.verbose = verbose
        self.output_dir = output_dir

        self.device = "cpu"
        if cuda and torch.cuda.is_available():
            self.device = "cuda"

        # build model
        self.model = Model(
            seq_len=self.seq_len,
            channels=self.input_size,
            cut_freq=self.cut_freq,
            pred_len=self.pred_len,
            individual=self.individual
        )
        self.optimizer = import_object(optimizer)(self.model.parameters(), lr=self.learning_rate)
        self.criterion = nn.MSELoss()
        self.model.to(self.device)

    def _get_energy(self, data_loader):
        self.model.eval()

        energy = []
        predictions = []
        for i, input_data in enumerate(data_loader):
            x = input_data.to(self.device)[:,::self.DSR,:]
            output, _ = self.model(x)
            predictions.append(output.detach().cpu().numpy())
            
            for u in range(output.shape[0]):
                rec_loss = self.criterion(output[u], input_data[u].to(self.device))
                cri = rec_loss.unsqueeze(0)
                cri = cri.detach().cpu().numpy()
                energy.append(cri)

        return np.concatenate(energy, axis=0), np.concatenate(predictions, axis=0)

    def _validate(self, valid_loader):
        self.model.eval()

        losses = list()
        for input_data in valid_loader:
            x = input_data.to(self.device)[:,::self.DSR,:]
            output, _ = self.model(x)

            rec_loss = self.criterion(output, input_data.to(self.device))

            losses.append(rec_loss.item())

        return np.mean(losses)

    def _fit(self, train_loader, valid_loader):
        for epoch in range(self.epochs):
            losses = []
            self.model.train()

            for input_data in train_loader:
                self.optimizer.zero_grad()
                x = input_data.to(self.device)[:,::self.DSR,:]
                output, _ = self.model(x)

                rec_loss = self.criterion(output, input_data.to(self.device))
                losses.append(rec_loss.item())

                rec_loss.backward(retain_graph=True)
                self.optimizer.step()

            if valid_loader is not None:
                valid_loss = self._validate(valid_loader)
            else:
                valid_loss = None

            if self.verbose:
                print('Epoch: {}/{}, Loss: {}, Valid Loss {}'.format(
                    epoch + 1, self.epochs, np.mean(losses), valid_loss))

            self._adjust_learning_rate(self.optimizer, epoch + 1, self.learning_rate)

    def fit(self, X):
        train = X
        valid_loader = None

        # split data
        if self.valid_split > 0:
            valid_size = int(len(X) * self.valid_split)
            train = X[: -valid_size]
            valid = X[-valid_size:]

            valid_loader = DataLoader(dataset=Signal(valid, self.window_size, self.step),
                                      batch_size=self.batch_size,
                                      shuffle=False)

        train_loader = DataLoader(dataset=Signal(train, self.window_size, self.step),
                                  batch_size=self.batch_size,
                                  shuffle=self.shuffle)

        self._fit(train_loader, valid_loader)

        if self.output_dir:
            model_dir = Path(self.output_dir)
            os.makedirs(model_dir, exist_ok=True)
            LOGGER.info(f"Saving model to {model_dir}.")
            torch.save(self.model.state_dict(), model_dir + f'checkpoint_{self.epochs}.pth')

        self.train_energy, train_predictions = self._get_energy(train_loader)

    def predict(self, X):
        data_loader = DataLoader(dataset=Signal(X, self.window_size, self.step, mode='test'),
                                 batch_size=self.batch_size)

        energy, predictions = self._get_energy(data_loader)
        return predictions, energy, self.train_energy
