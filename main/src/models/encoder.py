#!/usr/bin/env python
# -*- coding:utf-8 -*-

__author__ = 'Shining'
__email__ = 'mrshininnnnn@gmail.com'


# dependency
# public
import math
import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    """Positional Encoder for Transformer"""
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class GRURNNEncoder(nn.Module):
    """RNN Encoder with Gated Recurrent Unit (GRU)"""
    def __init__(self, config): 
        super(GRURNNEncoder, self).__init__()
        self.config = config
        self.embedding = nn.Embedding(
            num_embeddings=self.config.src_vocab_size, 
            embedding_dim=self.config.embedding_size, 
            padding_idx=self.config.pad_idx)
        self.em_dropout = nn.Dropout(self.config.embedding_drop_rate)
        self.gru = nn.GRU(
            input_size=self.config.embedding_size, 
            hidden_size=self.config.en_hidden_size, 
            num_layers=self.config.en_num_layers, 
            batch_first=True, 
            dropout=0, 
            bidirectional=False)
        self.gru_dropout = nn.Dropout(self.config.en_drop_rate)

    def forward(self, x, x_lens):
        # x: batch_size, max_xs_seq_len
        # batch_size, max_xs_seq_len, embedding_dim
        x = self.embedding(x)
        x = self.em_dropout(x)
        # batch_size*max_xs_seq_len, embedding_dim
        x = nn.utils.rnn.pack_padded_sequence(
            input=x, 
            lengths=x_lens, 
            batch_first=True, 
            enforce_sorted=False)
        # x: batch_size*max_xs_seq_len, embedding_dim
        # h: 1, batch_size, en_hidden_size
        x, h = self.gru(x)
        # x: batch_size, max_xs_seq_len, embedding_dim
        x, _ = nn.utils.rnn.pad_packed_sequence(
            sequence=x, 
            batch_first=True)
        # batch_size, max_xs_seq_len, en_hidden_size
        x = self.gru_dropout(x)

        return x, h


class LSTMRNNEncoder(nn.Module):
    """RNN Enocder with Long Short Term Memory (LSTM)"""
    def __init__(self, config):
        super(LSTMRNNEncoder, self).__init__()
        self.config = config
        self.embedding = nn.Embedding(
            num_embeddings=self.config.src_vocab_size, 
            embedding_dim=self.config.embedding_size, 
            padding_idx=self.config.pad_idx)
        self.em_dropout = nn.Dropout(self.config.embedding_drop_rate)
        self.lstm = nn.LSTM(
            input_size=self.config.embedding_size, 
            hidden_size=self.config.en_hidden_size, 
            num_layers=self.config.en_num_layers, 
            batch_first=True, 
            dropout=0, 
            bidirectional=False)
        self.lstm_dropout = nn.Dropout(self.config.en_drop_rate)

    def forward(self, x, x_lens):
        # x: batch_size
        # batch_size, max_xs_seq_len, embedding_dim
        x = self.embedding(x)
        x = self.em_dropout(x)
        # batch_size*max_xs_seq_len, embedding_dim
        x = nn.utils.rnn.pack_padded_sequence(
            input=x, 
            lengths=x_lens, 
            batch_first=True, 
            enforce_sorted=False)
        # x: batch_size*max_xs_seq_len, embedding_dim
        # hidden: (h, c)
        # h, c: 1, batch_size, en_hidden_size
        x, hidden = self.lstm(x)
        # x: batch_size, max_xs_seq_len, embedding_dim
        x, _ = nn.utils.rnn.pad_packed_sequence(
            sequence=x, 
            batch_first=True)
        # batch_size, max_xs_seq_len, en_hidden_size
        x = self.lstm_dropout(x)

        return x, hidden


class BiGRURNNEncoder(nn.Module):
    """Bidirectional RNN Enocder with Gated Recurrent Unit (GRU)"""
    def __init__(self, config):
        super(BiGRURNNEncoder, self).__init__()
        self.config = config
        self.embedding = nn.Embedding(
            num_embeddings=self.config.src_vocab_size, 
            embedding_dim=self.config.embedding_size, 
            padding_idx=self.config.pad_idx)
        self.em_dropout = nn.Dropout(self.config.embedding_drop_rate)
        self.bi_gru = nn.GRU(
            input_size=self.config.embedding_size, 
            hidden_size=self.config.en_hidden_size//2, 
            num_layers=self.config.en_num_layers, 
            batch_first=True, 
            dropout=0, 
            bidirectional=True)
        self.gru_dropout = nn.Dropout(self.config.en_drop_rate)

    def forward(self, x, x_lens):
        # x: batch_size, max_xs_seq_len
        # batch_size, max_xs_seq_len, embedding_dim
        x = self.embedding(x)
        x = self.em_dropout(x)
        # batch_size*max_xs_seq_len, embedding_dim
        x = nn.utils.rnn.pack_padded_sequence(
            input=x, 
            lengths=x_lens, 
            batch_first=True, 
            enforce_sorted=False)
        # x: batch_size*max_xs_seq_len, embedding_dim
        x, h = self.bi_gru(x)
        # batch_size, max_xs_seq_len, embedding_dim
        x, _ = nn.utils.rnn.pad_packed_sequence(
            sequence=x, 
            batch_first=True)
        # h: 1, batch_size, en_hidden_size
        h = torch.unsqueeze(torch.cat(torch.unbind(h, 0), 1), 0)
        x = self.gru_dropout(x)

        return x, h


class BiLSTMRNNEncoder(nn.Module):
    """Bidirectional RNN Enocder with Long Short Term Memory (LSTM)"""
    def __init__(self, config):
        super(BiLSTMRNNEncoder, self).__init__()
        self.config = config
        self.embedding = nn.Embedding(
            num_embeddings=self.config.src_vocab_size, 
            embedding_dim=self.config.embedding_size, 
            padding_idx=self.config.pad_idx)
        self.em_dropout = nn.Dropout(self.config.embedding_drop_rate)
        self.bi_lstm = nn.LSTM(
            input_size=self.config.embedding_size, 
            hidden_size=self.config.en_hidden_size//2, 
            num_layers=self.config.en_num_layers, 
            batch_first=True, 
            dropout=0, 
            bidirectional=True)
        self.lstm_dropout = nn.Dropout(self.config.en_drop_rate)

    def forward(self, x, x_lens):
        # x: batch_size, max_xs_seq_len
        # batch_size, max_xs_seq_len, embedding_dim
        x = self.embedding(x)
        x = self.em_dropout(x)
        # batch_size*max_xs_seq_len, embedding_dim
        x = nn.utils.rnn.pack_padded_sequence(
            input=x, 
            lengths=x_lens, 
            batch_first=True, 
            enforce_sorted=False)
        # x: batch_size*max_xs_seq_len, embedding_dim
        x, (h, c) = self.bi_lstm(x)
        # batch_size, max_xs_seq_len, embedding_dim
        x, _ = nn.utils.rnn.pad_packed_sequence(
            sequence=x, 
            batch_first=True)
        # h, c: 1, batch_size, en_hidden_size
        h = torch.unsqueeze(torch.cat(torch.unbind(h, 0), 1), 0)
        c = torch.unsqueeze(torch.cat(torch.unbind(c, 0), 1), 0)
        x = self.lstm_dropout(x)

        return x, (h, c)