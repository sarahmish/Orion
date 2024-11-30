"""Primitive Anomaly Transformer

This primitive is an pytorch implementation of "TimesNet: Temporal 
2D-Variation Modeling for General Time Series Analysis"
https://arxiv.org/pdf/2210.02186

This is a modified version of the original code, which can be found
at https://github.com/thuml/Time-Series-Library/blob/main/models/TimesNet.py
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
import torch.fft
from mlstars.utils import import_object
from torch.utils.data import DataLoader

from orion.primitives.anomaly_transformer import Signal, TokenEmbedding, PositionalEncoding
from orion.primitives.timeseries_anomalies import _merge_sequences, _prune_anomalies

LOGGER = logging.getLogger(__name__)


class FixedEmbedding(nn.Module):
    def __init__(self, c_in, d_model):
        super(FixedEmbedding, self).__init__()

        w = torch.zeros(c_in, d_model).float()
        w.require_grad = False

        position = torch.arange(0, c_in).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float()
                    * -(math.log(10000.0) / d_model)).exp()

        w[:, 0::2] = torch.sin(position * div_term)
        w[:, 1::2] = torch.cos(position * div_term)

        self.emb = nn.Embedding(c_in, d_model)
        self.emb.weight = nn.Parameter(w, requires_grad=False)

    def forward(self, x):
        return self.emb(x).detach()

class TemporalEmbedding(nn.Module):
    def __init__(self, d_model, embed_type='fixed', freq='h'):
        super(TemporalEmbedding, self).__init__()

        minute_size = 4
        hour_size = 24
        weekday_size = 7
        day_size = 32
        month_size = 13

        Embed = FixedEmbedding if embed_type == 'fixed' else nn.Embedding
        if freq == 't':
            self.minute_embed = Embed(minute_size, d_model)

        self.hour_embed = Embed(hour_size, d_model)
        self.weekday_embed = Embed(weekday_size, d_model)
        self.day_embed = Embed(day_size, d_model)
        self.month_embed = Embed(month_size, d_model)

    def forward(self, x):
        x = x.long()
        minute_x = self.minute_embed(x[:, :, 4]) if hasattr(
            self, 'minute_embed') else 0.
        hour_x = self.hour_embed(x[:, :, 3])
        weekday_x = self.weekday_embed(x[:, :, 2])
        day_x = self.day_embed(x[:, :, 1])
        month_x = self.month_embed(x[:, :, 0])

        return hour_x + weekday_x + day_x + month_x + minute_x

class TimeFeatureEmbedding(nn.Module):
    def __init__(self, d_model, embed_type='timeF', freq='h'):
        super(TimeFeatureEmbedding, self).__init__()

        freq_map = {'h': 4, 't': 5, 's': 6,
                    'm': 1, 'a': 1, 'w': 2, 'd': 3, 'b': 3}
        d_inp = freq_map[freq]
        self.embed = nn.Linear(d_inp, d_model, bias=False)

    def forward(self, x):
        return self.embed(x)

class DataEmbedding(nn.Module):
    def __init__(self, input_size, d_model, embed_type='fixed', freq='h', dropout=0.1):
        super(DataEmbedding, self).__init__()

        self.value_embedding = TokenEmbedding(input_size=input_size, d_model=d_model)
        self.position_embedding = PositionalEncoding(d_model=d_model)
        self.temporal_embedding = TemporalEmbedding(d_model=d_model, embed_type=embed_type,
                                                    freq=freq) if embed_type != 'timeF' else TimeFeatureEmbedding(
            d_model=d_model, embed_type=embed_type, freq=freq)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, x_mark):
        if x_mark is None:
            x = self.value_embedding(x) + self.position_embedding(x)
        else:
            x = self.value_embedding(
                x) + self.temporal_embedding(x_mark) + self.position_embedding(x)
        return self.dropout(x)
    

def FFT_for_Period(x, k=2):
    # [B, T, C]
    xf = torch.fft.rfft(x, dim=1)
    # find period by amplitudes
    frequency_list = abs(xf).mean(0).mean(-1)
    frequency_list[0] = 0
    _, top_list = torch.topk(frequency_list, k)
    top_list = top_list.detach().cpu().numpy()
    period = x.shape[1] // top_list
    return period, abs(xf).mean(-1)[:, top_list]


class Inception_Block_V1(nn.Module):
    def __init__(self, in_channels, out_channels, num_kernels=6, init_weight=True):
        super(Inception_Block_V1, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_kernels = num_kernels
        kernels = []
        for i in range(self.num_kernels):
            kernels.append(nn.Conv2d(in_channels, out_channels, kernel_size=2 * i + 1, padding=i))
        self.kernels = nn.ModuleList(kernels)
        if init_weight:
            self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        res_list = []
        for i in range(self.num_kernels):
            res_list.append(self.kernels[i](x))
        res = torch.stack(res_list, dim=-1).mean(-1)
        return res


class TimesBlock(nn.Module):
    def __init__(self, seq_len, pred_len, top_k, d_model, d_ff, num_kernels):
        super(TimesBlock, self).__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.k = top_k
        # parameter-efficient design
        self.conv = nn.Sequential(
            Inception_Block_V1(d_model, d_ff,
                               num_kernels=num_kernels),
            nn.GELU(),
            Inception_Block_V1(d_ff, d_model,
                               num_kernels=num_kernels)
        )

    def forward(self, x):
        B, T, N = x.size()
        period_list, period_weight = FFT_for_Period(x, self.k)

        res = []
        for i in range(self.k):
            period = period_list[i]
            # padding
            if (self.seq_len + self.pred_len) % period != 0:
                length = (
                                 ((self.seq_len + self.pred_len) // period) + 1) * period
                padding = torch.zeros([x.shape[0], (length - (self.seq_len + self.pred_len)), x.shape[2]]).to(x.device)
                out = torch.cat([x, padding], dim=1)
            else:
                length = (self.seq_len + self.pred_len)
                out = x
            # reshape
            out = out.reshape(B, length // period, period,
                              N).permute(0, 3, 1, 2).contiguous()
            # 2D conv: from 1d Variation to 2d Variation
            out = self.conv(out)
            # reshape back
            out = out.permute(0, 2, 3, 1).reshape(B, -1, N)
            res.append(out[:, :(self.seq_len + self.pred_len), :])
        res = torch.stack(res, dim=-1)
        # adaptive aggregation
        period_weight = F.softmax(period_weight, dim=1)
        period_weight = period_weight.unsqueeze(
            1).unsqueeze(1).repeat(1, T, N, 1)
        res = torch.sum(res * period_weight, -1)
        # residual connection
        res = res + x
        return res


class Model(nn.Module):
    def __init__(self, input_size=1, output_size=1, window_size=100, pred_len=0, top_k=5, 
                 d_model=512, d_ff=2048, num_kernels=6, e_layers=2, embed='timeF', 
                 freq='h', dropout=0.1):
        super(Model, self).__init__()

        self.window_size = window_size
        self.pred_len = pred_len
        self.top_k = top_k
        self.d_model = d_model
        self.d_ff = d_ff
        self.num_kernels = num_kernels
        self.layers = e_layers
        self.input_size = input_size
        self.embed = embed
        self.freq = freq
        self.dropout = dropout
        self.output_size = output_size
        self.model = nn.ModuleList([
            TimesBlock(
                self.window_size, 
                self.pred_len, 
                self.top_k, 
                self.d_model, 
                self.d_ff, 
                self.num_kernels)
            for _ in range(self.layers)
        ])
        
        self.enc_embedding = DataEmbedding(self.input_size, self.d_model, self.embed, self.freq,
                                           self.dropout)
        
        self.layer_norm = nn.LayerNorm(self.d_model)
        self.predict_linear = nn.Linear(
            self.window_size, self.pred_len + self.window_size)
        self.projection = nn.Linear(
            self.d_model, self.output_size, bias=True)

    def anomaly_detection(self, x_enc):
        # Normalization from Non-stationary Transformer
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(
            torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        
        x_enc /= stdev

        # embedding
        enc_out = self.enc_embedding(x_enc, None)  # [B,T,C]
        # TimesNet
        for i in range(self.layers):
            enc_out = self.layer_norm(self.model[i](enc_out))
        # project back
        dec_out = self.projection(enc_out)

        # De-Normalization from Non-stationary Transformer
        dec_out = dec_out * \
                  (stdev[:, 0, :].unsqueeze(1).repeat(
                      1, self.pred_len + self.window_size, 1))
        dec_out = dec_out + \
                  (means[:, 0, :].unsqueeze(1).repeat(
                      1, self.pred_len + self.window_size, 1))
        return dec_out


    def forward(self, x_enc, mask=None):
        dec_out = self.anomaly_detection(x_enc)
        return dec_out  # [B, L, D]


class TimesNet():
    """TimesNet model for unsupervised time series anomaly detection.

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

    def __init__(self, input_size=1, output_size=1, window_size=100, pred_len=0, step=1, 
                 top_k=3, d_model=512, d_ff=2048, num_kernels=6, embed='fixed', freq='h',
                 features='M', dropout=0.0, num_layers=2, batch_size=256, 
                 learning_rate=1e-4, epochs=10, valid_split=0.0, shuffle=True, 
                 cuda=True, optimizer="torch.optim.Adam", verbose=False, output_dir=False):

        self.window_size = window_size
        self.pred_len = pred_len
        self.step = step
        self.top_k = top_k
        self.d_model = d_model
        self.d_ff = d_ff
        self.num_kernels = num_kernels
        self.num_layers = num_layers
        self.input_size = input_size
        self.embed = embed
        self.freq = freq
        self.dropout = dropout
        self.output_size = output_size
        self.features = features

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
            input_size=self.input_size,
            output_size=self.output_size,
            window_size=self.window_size,
            pred_len=self.pred_len,
            top_k=self.top_k,
            d_model=self.d_model,
            d_ff=self.d_ff,
            num_kernels=self.num_kernels,
            e_layers=self.num_layers,
            embed=self.embed,
            freq=self.freq,
            dropout=self.dropout,
        )

        self.optimizer = import_object(optimizer)(self.model.parameters(), lr=self.learning_rate)
        self.criterion = nn.MSELoss()
        self.mse = nn.MSELoss(reduce=False)
        self.model.to(self.device)

    def _get_energy(self, data_loader):
        self.model.eval()

        energy = []
        predictions = []
        for i, input_data in enumerate(data_loader):
            x = input_data.to(self.device)
            output = self.model(x)
            
            score = torch.mean(self.mse(x, output), dim=-1)
            score = score.detach().cpu().numpy()
            energy.append(score)
            
            predictions.append(output.detach().cpu().numpy())

        return np.concatenate(energy, axis=0), np.concatenate(predictions, axis=0)

    def _validate(self, valid_loader):
        self.model.eval()

        losses = list()
        for input_data in valid_loader:
            x = input_data.to(self.device)
            output = self.model(x)

            f_dim = -1 if self.features == 'MS' else 0
            output = output[:, :, f_dim:]
            pred = output.detach().cpu()
            true = x.detach().cpu()

            loss = self.criterion(pred, true)
            losses.append(loss.item())

        return np.mean(losses)

    def _fit(self, train_loader, valid_loader):
        for epoch in range(self.epochs):
            losses = []
            self.model.train()

            for input_data in train_loader:
                self.optimizer.zero_grad()
                x = input_data.to(self.device)
                output = self.model(x)

                f_dim = -1 if self.features == 'MS' else 0
                output = output[:, :, f_dim:]

                loss = self.criterion(output, x)
                losses.append(loss.item())

                loss.backward()
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
