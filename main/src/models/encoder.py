#!/usr/bin/env python
# -*- coding:utf-8 -*-
__author__ = 'Shining'
__email__ = 'mrshininnnnn@gmail.com'


# built-in
import math
# public
import torch
import torch.nn as nn
import torch.nn.functional as F


class BiLSTMRNNEncoder(nn.Module):
    """Bidirectional RNN Enocder with Long Short Term Memory (LSTM)"""
    def __init__(self, config):
        super(BiLSTMRNNEncoder, self).__init__()
        self.config = config
        self.embedding = nn.Embedding(
            num_embeddings=self.config.src_vocab_size, 
            embedding_dim=self.config.embedding_size, 
            padding_idx=self.config.PAD_IDX)
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
            lengths=x_lens.to('cpu'), 
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

# adapted from https://github.com/bentrevett/pytorch-seq2seq/
class CNNEncoder(nn.Module):
    def __init__(self, 
                 input_dim, 
                 emb_dim, 
                 hid_dim, 
                 n_layers, 
                 kernel_size, 
                 dropout, 
                 device,
                 max_length = 100):
        super().__init__()
        
        assert kernel_size % 2 == 1, "Kernel size must be odd!"
        
        self.device = device
        
        self.scale = torch.sqrt(torch.FloatTensor([0.5])).to(device)
        
        self.tok_embedding = nn.Embedding(input_dim, emb_dim)
        self.pos_embedding = nn.Embedding(max_length, emb_dim)
        
        self.emb2hid = nn.Linear(emb_dim, hid_dim)
        self.hid2emb = nn.Linear(hid_dim, emb_dim)
        
        self.convs = nn.ModuleList([nn.Conv1d(in_channels = hid_dim, 
                                              out_channels = 2 * hid_dim, 
                                              kernel_size = kernel_size, 
                                              padding = (kernel_size - 1) // 2)
                                    for _ in range(n_layers)])
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, src):
        
        #src = [batch size, src len]
        
        batch_size = src.shape[0]
        src_len = src.shape[1]
        
        #create position tensor
        pos = torch.arange(0, src_len).unsqueeze(0).repeat(batch_size, 1).to(self.device)
        
        #pos = [0, 1, 2, 3, ..., src len - 1]
        
        #pos = [batch size, src len]
        
        #embed tokens and positions
        tok_embedded = self.tok_embedding(src)
        pos_embedded = self.pos_embedding(pos)
        
        #tok_embedded = pos_embedded = [batch size, src len, emb dim]
        
        #combine embeddings by elementwise summing
        embedded = self.dropout(tok_embedded + pos_embedded)
        
        #embedded = [batch size, src len, emb dim]
        
        #pass embedded through linear layer to convert from emb dim to hid dim
        conv_input = self.emb2hid(embedded)
        
        #conv_input = [batch size, src len, hid dim]
        
        #permute for convolutional layer
        conv_input = conv_input.permute(0, 2, 1) 
        
        #conv_input = [batch size, hid dim, src len]
        
        #begin convolutional blocks...
        
        for i, conv in enumerate(self.convs):
        
            #pass through convolutional layer
            conved = conv(self.dropout(conv_input))

            #conved = [batch size, 2 * hid dim, src len]

            #pass through GLU activation function
            conved = F.glu(conved, dim = 1)

            #conved = [batch size, hid dim, src len]
            
            #apply residual connection
            conved = (conved + conv_input) * self.scale

            #conved = [batch size, hid dim, src len]
            
            #set conv_input to conved for next loop iteration
            conv_input = conved
        
        #...end convolutional blocks
        
        #permute and convert back to emb dim
        conved = self.hid2emb(conved.permute(0, 2, 1))
        
        #conved = [batch size, src len, emb dim]
        
        #elementwise sum output (conved) and input (embedded) to be used for attention
        combined = (conved + embedded) * self.scale
        
        #combined = [batch size, src len, emb dim]
        
        return conved, combined