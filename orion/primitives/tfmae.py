"""Primitive Anomaly Transformer

This primitive is an pytorch implementation of "Temporal-Frequency
Masked Autoencoders for Time Series Anomaly Detection"
https://github.com/LMissher/TFMAE/blob/main/paper/TFMAE.pdf

This is a modified version of the original code, which can be found
at https://github.com/LMissher/TFMAE/tree/main
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
from tqdm import tqdm
from mlstars.utils import import_object
from torch.utils.data import DataLoader

from orion.primitives.timeseries_anomalies import _merge_sequences, _prune_anomalies

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
        if self.mode == 'test':
            return (self.data.shape[0] - self.window_size) // self.window_size + 1

        return (self.data.shape[0] - self.window_size) // self.step + 1

    def __getitem__(self, index):
        start = index * self.step
        end = start + self.window_size

        if self.mode == 'train':
            return np.float32(self.data[start: end])
        elif self.mode == 'test':
            start = start // self.step * self.window_size
            return np.float32(self.data[start: start + self.window_size])
        else:
            raise ValueError(f'Unknown {self.mode} mode.')


class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEmbedding, self).__init__()
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe
        self.register_buffer('pe', pe)

    def forward(self, data=None, idx=None):
        if data != None:
            p = self.pe[:data].unsqueeze(0)
        else:
            p = self.pe.unsqueeze(0).repeat(idx.shape[0],1,1)[torch.arange(idx.shape[0])[:,None],idx,:]
        return p


class TokenEmbedding(nn.Module):
    def __init__(self, c_in, d_model):
        super(TokenEmbedding, self).__init__()
        padding = 1 if torch.__version__ >= '1.5.0' else 2
        self.tokenConv = nn.Conv1d(in_channels=c_in, out_channels=d_model,
                                   kernel_size=3, padding=padding, padding_mode='circular', bias=False)
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu')

    def forward(self, x):
        x = self.tokenConv(x.permute(0, 2, 1)).transpose(1, 2)
        return x


class DataEmbedding(nn.Module):
    def __init__(self, c_in, d_model, dropout=0.05):
        super(DataEmbedding, self).__init__()

        self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model)
        self.position_embedding = PositionalEmbedding(d_model=d_model)

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        x = self.value_embedding(x) + self.position_embedding(data = x.shape[1])
        return self.dropout(x)
    

class AttentionLayer(nn.Module):
    def __init__(self, d_model):
        super(AttentionLayer, self).__init__()

        self.norm = nn.LayerNorm(d_model)

        self.query_projection = nn.Linear(d_model, d_model)
        self.key_projection = nn.Linear(d_model, d_model)
        self.value_projection = nn.Linear(d_model, d_model)
        self.out_projection = nn.Linear(d_model, d_model)

    def forward(self, x):
        # [B, T, D]
        B, T, D = x.shape

        queries = self.query_projection(x)
        keys = self.key_projection(x).transpose(1,2)
        values = self.value_projection(x)

        attn = torch.softmax(torch.matmul(queries, keys) / math.sqrt(D), -1)

        out = torch.matmul(attn, values) + x

        return self.out_projection(self.norm(out)) + out, attn
    
class Encoder(nn.Module):
    def __init__(self, attn_layers, norm_layer=None):
        super(Encoder, self).__init__()
        self.attn_layers = nn.ModuleList(attn_layers)
        self.norm = norm_layer

    def forward(self, x, attn_mask=None):
        # x [B, T, D]
        attlist = []
        for attn_layer in self.attn_layers:
            x, _ = attn_layer(x)
            attlist.append(_)

        if self.norm is not None:
            x = self.norm(x)

        return x, attlist

class FreEnc(nn.Module):
    def __init__(self, c_in, c_out, d_model, e_layers, win_size, fr):
        super(FreEnc, self).__init__()

        self.emb = DataEmbedding(c_in, d_model)

        self.enc = Encoder(
            [
                    AttentionLayer(d_model) for l in range(e_layers)
            ],
            norm_layer=nn.LayerNorm(d_model)
        )

        self.pro = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
            nn.Sigmoid()
        )

        self.mask_token = nn.Parameter(torch.zeros(1,d_model,1, dtype=torch.cfloat))

        self.fr = fr
    
    def forward(self, x):
        # x: [B, T, C]
        ex = self.emb(x) # [B, T, D]

        # converting to frequency domain and calculating the mag
        cx = torch.fft.rfft(ex.transpose(1,2))
        mag = torch.sqrt(cx.real ** 2 + cx.imag ** 2) # [B, D, Mag]

        # masking smaller mag
        quantile = torch.quantile(mag, self.fr, dim=2, keepdim=True)
        idx = torch.argwhere(mag<quantile)
        cx[mag<quantile] = self.mask_token.repeat(ex.shape[0], 1, mag.shape[-1])[idx[:,0],idx[:,1],idx[:,2]]

        # converting to time domain
        ix = torch.fft.irfft(cx).transpose(1,2)

        # encoding tokens
        dx, att = self.enc(ix)

        rec = self.pro(dx)
        att.append(rec)

        return att # att(list): [B, T, T]
    

class TemEnc(nn.Module):
    def __init__(self, c_in, c_out, d_model, e_layers, win_size, seq_size, tr):
        super(TemEnc, self).__init__()

        self.emb = DataEmbedding(c_in, d_model)
        self.pos_emb = PositionalEmbedding(d_model)

        self.enc = Encoder(
            [
                    AttentionLayer(d_model) for l in range(e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(d_model)
        )

        self.dec = Encoder(
            [
                    AttentionLayer(d_model) for l in range(e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(d_model)
        )

        self.pro = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
            nn.Sigmoid()
        )

        self.mask_token = nn.Parameter(torch.zeros(1,1,d_model))
        self.tr = int(tr * win_size)
        self.seq_size = seq_size
    
    def forward(self, x):
        # x: [B, T, C]
        ex = self.emb(x) # [B, T, D]
        filters = torch.ones(1,1,self.seq_size).to(device)
        ex2 = ex ** 2

        # calculating summation of ex and ex2
        ltr = F.conv1d(ex.transpose(1,2).reshape(-1, ex.shape[1]).unsqueeze(1), filters, padding=self.seq_size-1)
        ltr[:,:,:self.seq_size-1] /= torch.arange(1,self.seq_size).to(device)
        ltr[:,:,self.seq_size-1:] /= self.seq_size
        ltr2 = F.conv1d(ex2.transpose(1,2).reshape(-1, ex.shape[1]).unsqueeze(1), filters, padding=self.seq_size-1)
        ltr2[:,:,:self.seq_size-1] /= torch.arange(1,self.seq_size).to(device)
        ltr2[:,:,self.seq_size-1:] /= self.seq_size
        
        # calculating mean and variance
        ltrd = (ltr2 - ltr ** 2)[:,:,:ltr.shape[-1]-self.seq_size+1].squeeze(1).reshape(ex.shape[0],ex.shape[-1],-1).transpose(1,2)
        ltrm = ltr[:,:,:ltr.shape[-1]-self.seq_size+1].squeeze(1).reshape(ex.shape[0],ex.shape[-1],-1).transpose(1,2)
        score = ltrd.sum(-1) / ltrm.sum(-1)

        # mask time points
        masked_idx, unmasked_idx = score.topk(self.tr, dim=1, sorted=False)[1], (-1*score).topk(x.shape[1]-self.tr, dim=1, sorted=False)[1]
        unmasked_tokens = ex[torch.arange(ex.shape[0])[:,None],unmasked_idx,:]
        
        # encoding unmasked tokens and getting masked tokens
        ux, _ = self.enc(unmasked_tokens)
        masked_tokens = self.mask_token.repeat(ex.shape[0], masked_idx.shape[1], 1) + self.pos_emb(idx = masked_idx)
        
        tokens = torch.zeros(ex.shape,device=device)
        tokens[torch.arange(ex.shape[0])[:,None],unmasked_idx,:] = ux
        tokens[torch.arange(ex.shape[0])[:,None],masked_idx,:] = masked_tokens

        # decoding tokens
        dx, att = self.dec(tokens)

        rec = self.pro(dx)
        att.append(rec)

        return att # att(list): [B, T, T]


class MTFA(nn.Module):
    def __init__(self, win_size, seq_size, c_in, c_out, d_model=512, e_layers=3, fr=0.4, tr=0.5, dev=None):
        super(MTFA, self).__init__()
        global device
        device = dev
        self.tem = TemEnc(c_in, c_out, d_model, e_layers, win_size, seq_size, tr)
        self.fre = FreEnc(c_in, c_out, d_model, e_layers, win_size, fr)

    def forward(self, x):
        # x: [B, T, C]
        tematt = self.tem(x) # tematt: [B, T, T]
        freatt = self.fre(x) # freatt: [B, T, T]
        return tematt, freatt
    

class TFMAE():
    @staticmethod
    def _kl_loss(p, q):
        res = p * (torch.log(p + 0.0001) - torch.log(q + 0.0001))
        return torch.mean(torch.sum(res, dim=-1), dim=1)

    @staticmethod
    def _adjust_learning_rate(optimizer, epoch, lr_):
        lr_adjust = {epoch: lr_ * (0.5 ** ((epoch - 1) // 1))}
        if epoch in lr_adjust.keys():
            lr = lr_adjust[epoch]
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

            LOGGER.info(f'Updating learning rate to {lr}')

    def __init__(self, input_size=1, output_size=1, window_size=100, seq_size=10, step=1, k=3,
                 d_model=512, fr=0.4, tr=0.5,
                 n_hidden=512, num_layers=3, batch_size=256, learning_rate=1e-4,
                 temperature=50, epochs=10, valid_split=0.2, shuffle=True, cuda=True,
                 optimizer="torch.optim.Adam", verbose=False, output_dir=False):
        
        self.input_size = input_size
        self.output_size = output_size
        self.window_size = window_size
        self.seq_size = seq_size
        self.step = step

        self.k = k
        self.d_model = d_model
        self.fr = fr
        self.tr = tr
        self.n_hidden = n_hidden
        self.num_layers = num_layers

        self.temperature = temperature
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


        self.model = MTFA(
            win_size=self.window_size, 
            seq_size=self.seq_size, 
            c_in=self.input_size, 
            c_out=self.output_size, 
            d_model=self.d_model, 
            e_layers=self.num_layers, 
            fr=self.fr, 
            tr=self.tr, 
            dev=self.device
        )

        self.optimizer = import_object(optimizer)(self.model.parameters(), lr=self.learning_rate)
        self.criterion = nn.MSELoss()
        self.mse = nn.MSELoss(reduce=False)
        self.model.to(self.device)

    def _validate(self, valid_loader):
        self.model.eval()

        loss_list = []
        with torch.no_grad():
            for i, input_data in enumerate(valid_loader):
                input = input_data.float().to(self.device)

                tematt, freatt = self.model(input)

                adv_loss = 0.0
                con_loss = 0.0
                for u in range(len(freatt)):
                    adv_loss += (torch.mean(
                        self._kl_loss(tematt[u], (
                            freatt[u] / torch.unsqueeze(torch.sum(freatt[u], dim=-1), dim=-1)).detach())) + torch.mean(
                        self._kl_loss(
                            (freatt[u] / torch.unsqueeze(torch.sum(freatt[u], dim=-1), dim=-1)).detach(),
                            tematt[u])))
                    con_loss += (torch.mean(
                        self._kl_loss((freatt[u] / torch.unsqueeze(torch.sum(freatt[u], dim=-1), dim=-1)),
                                tematt[u].detach())) + torch.mean(
                        self._kl_loss(tematt[u].detach(),
                                (freatt[u] / torch.unsqueeze(torch.sum(freatt[u], dim=-1), dim=-1)))))

                adv_loss = adv_loss / len(freatt)
                con_loss = con_loss / len(freatt)

                loss_list.append((con_loss - adv_loss).item())

        return np.average(loss_list)
    
    def train(self, train_loader, valid_loader):
        train_steps = len(train_loader)
        for epoch in tqdm(range(self.epochs)):
            loss_list = []

            self.model.train()
            with tqdm(total=train_steps) as pbar:
                for i, input_data in enumerate(train_loader):

                    self.optimizer.zero_grad()

                    input = input_data.float().to(self.device)

                    tematt, freatt = self.model(input)

                    adv_loss = 0.0
                    con_loss = 0.0

                    for u in range(len(freatt)):
                        adv_loss += (torch.mean(
                            self._kl_loss(tematt[u], (
                                freatt[u] / torch.unsqueeze(torch.sum(freatt[u], dim=-1), dim=-1)).detach())) + torch.mean(
                            self._kl_loss((freatt[u] / torch.unsqueeze(torch.sum(freatt[u], dim=-1), dim=-1)).detach(),
                                    tematt[u])))
                        con_loss += (torch.mean(self._kl_loss(
                            (freatt[u] / torch.unsqueeze(torch.sum(freatt[u], dim=-1), dim=-1)),
                            tematt[u].detach())) + torch.mean(
                            self._kl_loss(tematt[u].detach(), (
                                    freatt[u] / torch.unsqueeze(torch.sum(freatt[u], dim=-1), dim=-1)))))

                    adv_loss = adv_loss / len(freatt)
                    con_loss = con_loss / len(freatt)

                    loss =  con_loss - adv_loss
                    loss_list.append(loss.item())

                    pbar.update(1)

                    loss.backward()
                    self.optimizer.step()

            train_loss = np.average(loss_list)
            vali_loss = self._validate(valid_loader)

            print(
                "Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} ".format(
                    epoch + 1, train_steps, train_loss, vali_loss))

    def _energy(self, data_loader):
        energy = []
        with torch.no_grad():
            for i, input_data in enumerate(data_loader):
                input = input_data.float().to(self.device)

                tematt, freatt = self.model(input)
                adv_loss = 0.0
                con_loss = 0.0
                for u in range(len(freatt)):
                    if u == 0:
                        adv_loss = self._kl_loss(tematt[u], (
                                freatt[u] / torch.unsqueeze(torch.sum(freatt[u], dim=-1), dim=-1)).detach()) * self.temperature
                        con_loss = self._kl_loss(
                            (freatt[u] / torch.unsqueeze(torch.sum(freatt[u], dim=-1), dim=-1)),
                            tematt[u].detach()) * self.temperature
                    else:
                        adv_loss += self._kl_loss(tematt[u], (
                                freatt[u] / torch.unsqueeze(torch.sum(freatt[u], dim=-1), dim=-1)).detach()) * self.temperature
                        con_loss += self._kl_loss(
                            (freatt[u] / torch.unsqueeze(torch.sum(freatt[u], dim=-1), dim=-1)),
                            tematt[u].detach()) * self.temperature

                metric = torch.softmax((adv_loss + con_loss), dim=-1)
                cri = metric.detach().cpu().numpy()
                energy.append(cri)

        energy = np.concatenate(energy, axis=0)
        return energy

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

        self.train(train_loader, valid_loader)

        if self.output_dir:
            model_dir = Path(self.output_dir)
            os.makedirs(model_dir, exist_ok=True)
            LOGGER.info(f"Saving model to {model_dir}.")
            torch.save(self.model.state_dict(), model_dir + f'checkpoint_{self.epochs}.pth')

        self.valid_energy = self._energy(valid_loader)


    def predict(self, X):
        data_loader = DataLoader(dataset=Signal(X, self.window_size, self.step, mode='test'),
                                 batch_size=self.batch_size)

        energy = self._energy(data_loader)
        return energy, self.valid_energy

