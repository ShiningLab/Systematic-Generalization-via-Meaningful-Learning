#!/usr/bin/env python
# -*- coding:utf-8 -*-

__author__ = 'Shining'
__email__ = 'mrshininnnnn@gmail.com'


# dependency
# public
import torch
import torch.nn as nn
import torch.nn.functional as F


class GRUPtrNetDecoderAttention(nn.Module):
    """docstring for GRUPtrNetDecoderAttention"""
    def __init__(self, config):
        super(GRUPtrNetDecoderAttention, self).__init__()
        self.config = config
        self.w1 = nn.Linear(config.en_hidden_size, config.de_hidden_size, bias=False)
        self.w2 = nn.Linear(config.de_hidden_size, config.de_hidden_size, bias=False)
        self.vt = nn.Linear(config.de_hidden_size, 1, bias=False)

    def forward(self, h, encoder_output, src_lens):
        # h: 1, batch_size, de_hidden_size/en_hidden_size
        # encoder_output: batch_size, max_seq_len, en_hidden_size
        # src_lens: batch_size
        # batch_size, max_seq_len, de_hidden_size 
        encoder_output = self.w1(encoder_output)
        # batch_size, 1, de_hidden_size
        h = self.w2(h).transpose(0, 1)
        # batch_size, max_seq_len
        u = self.vt(torch.tanh(encoder_output + h)).squeeze(-1)
        # max_seq_len
        idx = torch.arange(end=encoder_output.shape[1], dtype=torch.float, device=self.config.device)
        # batch_size, max_src_seq_len
        idx = idx.unsqueeze(0).expand(u.shape)
        # batch size, max_src_seq_len
        src_lens = src_lens.unsqueeze(-1).expand(u.shape)
        # batch_size, max_src_seq_len
        mask = idx < src_lens
        u.masked_fill(~mask, float('-inf'))
        # batch_size, max_src_seq_len
        return F.softmax(u, dim=1).unsqueeze(1)


class LSTMPtrNetDecoderAttention(nn.Module):
    """docstring for LSTMPtrNetDecoderAttention"""
    def __init__(self, config):
        super(LSTMPtrNetDecoderAttention, self).__init__()
        self.config = config
        self.w1 = nn.Linear(config.en_hidden_size, config.de_hidden_size, bias=False)
        self.w2 = nn.Linear(config.de_hidden_size*2, config.de_hidden_size, bias=False)
        self.vt = nn.Linear(config.de_hidden_size, 1, bias=False)

    def forward(self, h, encoder_output, src_lens):
        # h: batch_size, de_hidden_size/en_hidden_size
        # encoder_output: batch_size, max_seq_len, en_hidden_size
        # src_lens: batch_size
        # batch_size, max_seq_len, de_hidden_size 
        encoder_output = self.w1(encoder_output)
        # batch_size, 1, de_hidden_size
        h = self.w2(torch.cat(h, -1)).transpose(0, 1)
        # batch_size, max_seq_len
        # u = self.vt(torch.tanh(encoder_output + h)).squeeze(-1)
        # max_seq_len
        idx = torch.arange(end=encoder_output.shape[1], dtype=torch.float, device=self.config.device)
        # batch_size, max_src_seq_len
        idx = idx.unsqueeze(0).expand(u.shape)
        # batch size, max_src_seq_len
        src_lens = src_lens.unsqueeze(-1).expand(u.shape)
        # batch_size, max_src_seq_len
        mask = idx < src_lens
        u.masked_fill(~mask, float('-inf'))
        # batch_size, max_src_seq_len
        return F.softmax(u, dim=1).unsqueeze(1)

class GRURNNDecoderAttention(nn.Module):
    """docstring for GRURNNDecoderAttention"""
    def __init__(self, config):
        super(GRURNNDecoderAttention, self).__init__()
        self.config = config
        self.attn = torch.nn.Linear(
            self.config.de_hidden_size+self.config.en_hidden_size, 
            self.config.de_hidden_size)
        self.v = torch.nn.Parameter(torch.rand(self.config.de_hidden_size))
        self.v.data.normal_(mean=0, std=1./self.v.size(0)**(1./2.))

    def score(self, h, encoder_output):
        # h: batch_size, max_src_seq_len, de_hidden_size
        # encoder_output: batch_size, max_src_seq_len, en_hidden_size
        # batch_size, max_src_seq_len, de_hidden_size
        energy = torch.tanh(self.attn(torch.cat([h, encoder_output], 2)))
        # batch_size, de_hidden_size, max_src_seq_len
        energy = energy.transpose(2, 1)
        # batch_size, 1, de_hidden_size
        v = self.v.repeat(encoder_output.shape[0], 1).unsqueeze(1)
        # batch_size, 1, max_src_seq_len
        energy = torch.bmm(v, energy)
        # batch_size, max_src_seq_len
        return energy.squeeze(1)

    def forward(self, h, encoder_output, src_lens):
        # h: 1, batch_size, de_hidden_size
        # encoder_output: batch_size, max_src_seq_len, en_hidden_size
        # src_lens: batch_size
        # batch_size, en_hidden_size/de_hidden_size
        h = h.squeeze(0)
        # batch_size, max_src_seq_len, en_hidden_size/de_hidden_size
        h = h.repeat(encoder_output.shape[1], 1, 1).transpose(0, 1)
        # batch_size, max_src_seq_len
        attn_energies = self.score(h, encoder_output)
        # max_src_seq_len
        idx = torch.arange(end=encoder_output.shape[1], dtype=torch.float, device=self.config.device)
        # batch_size, max_src_seq_len
        idx = idx.unsqueeze(0).expand(attn_energies.shape)
        # batch size, max_src_seq_len
        src_lens = src_lens.unsqueeze(-1).expand(attn_energies.shape)
        mask = idx < src_lens
        attn_energies[~mask] = float('-inf') 
        # batch_size, 1, max_src_seq_len
        return F.softmax(attn_energies, dim=1).unsqueeze(1)


class LSTMRNNDecoderAttention(nn.Module):
    """docstring for LSTMRNNDecoderAttention"""
    def __init__(self, config):
        super(LSTMRNNDecoderAttention, self).__init__()
        self.config = config
        self.attn = torch.nn.Linear(
            self.config.de_hidden_size*2+self.config.en_hidden_size, 
            self.config.de_hidden_size)
        self.v = torch.nn.Parameter(torch.rand(self.config.de_hidden_size))
        self.v.data.normal_(mean=0, std=1./self.v.size(0)**(1./2.))

    def score(self, hidden, encoder_output):
        # hidden: batch_size, max_src_seq_len, de_hidden_size*2
        # encoder_output: batch_size, max_src_seq_len, en_hidden_size
        # batch_size, max_src_seq_len, de_hidden_size
        energy = torch.tanh(self.attn(torch.cat([hidden, encoder_output], 2)))
        # batch_size, de_hidden_size, max_src_seq_len
        energy = energy.transpose(2, 1)
        # batch_size, 1, de_hidden_size
        v = self.v.repeat(encoder_output.shape[0], 1).unsqueeze(1)
        # batch_size, 1, max_src_seq_len
        energy = torch.bmm(v, energy)
        # batch_size, max_src_seq_len
        return energy.squeeze(1)

    def forward(self, hidden, encoder_output, src_lens):
        # hidden: (h, c)
        # h, c: 1, batch_size, de_hidden_size
        # encoder_output: batch_size, max_src_seq_len, en_hidden_size
        # src_lens: batch_size
        # 1, batch_size, de_hidden_size*2
        hidden = torch.cat(hidden, 2).squeeze(0)
        # batch_size, max_src_seq_len, de_hidden_size*2
        hidden = hidden.repeat(encoder_output.shape[1], 1, 1).transpose(0, 1)
        # batch_size, max_src_seq_len
        attn_energies = self.score(hidden, encoder_output)
        # max_src_seq_len
        idx = torch.arange(end=encoder_output.shape[1], dtype=torch.float, device=self.config.device)
        # batch_size, max_src_seq_len
        idx = idx.unsqueeze(0).expand(attn_energies.shape)
        # batch size, max_src_seq_len
        src_lens = src_lens.unsqueeze(-1).expand(attn_energies.shape)
        mask = idx < src_lens
        attn_energies[~mask] = float('-inf') 
        # batch_size, 1, max_src_seq_len
        return F.softmax(attn_energies, dim=1).unsqueeze(1)