#!/usr/bin/env python
# -*- coding:utf-8 -*-

__author__ = 'Shining'
__email__ = 'mrshininnnnn@gmail.com'


# dependency
# public
import torch
import torch.nn as nn
import random
# private
from .encoder import BiGRURNNEncoder
from .decoder import GRUPtrNetDecoder


class End2EndModelGraph(nn.Module): 
    """docstring for End2EndModelGraph""" 
    def __init__(self, config): 
        super(End2EndModelGraph, self).__init__() 
        self.config = config
        self.encoder = BiGRURNNEncoder(config)
        self.decoder = GRUPtrNetDecoder(config)
        self.embedding = nn.Embedding(
            num_embeddings=2, 
            embedding_dim=self.config.embedding_size, 
            padding_idx=self.config.pad_idx)

    def forward(self, xs, x_lens, argsort_xs, teacher_forcing_ratio=0.5):
        # xs: batch_size, max_len
        # argsoft_xs: batch_size, max_len
        batch_size = xs.shape[0]
        # for ptr net, x_len = y_len
        max_len = xs.shape[1]
        # encoder_output: batch_size, max_len, en_hidden_size
        # encoder_hidden: 1, batch_size, en_hidden_size
        encoder_output, decoder_hidden = self.encoder(xs, x_lens)
        # batch_size
        decoder_input = torch.empty(
            batch_size, 
            dtype=torch.int64, 
            device=self.config.device)
        decoder_input.fill_(self.config.start_idx)
        # batch_size, embedding_dim
        decoder_input = self.embedding(decoder_input)
        # max_len, batch_size, max_len
        decoder_outputs = torch.zeros(max_len, batch_size, max_len, 
            device=self.config.device)
        for i in range(max_len):
            # decoder_hidden: 1, batch_size, de_hidden_size
            # attn_w: batch_size, max_seq_len
            decoder_hidden, attn_w = self.decoder(
                decoder_input, decoder_hidden, encoder_output, x_lens)
            # batch_size, max_seq_len
            decoder_outputs[i] = attn_w.log()
            # batch_size, 1
            ptr_idxes = argsort_xs[:, i, None] if random.random() < teacher_forcing_ratio \
            else attn_w.max(-1, keepdim=True)[1]
            # batch_size, 1, en_hidden_size
            ptr_idxes = ptr_idxes.unsqueeze(-1).expand(-1, 1, self.config.en_hidden_size)
            # batch_size, en_hidden_size
            decoder_input = torch.gather(encoder_output, dim=1, index=ptr_idxes).squeeze(1)

        # batch_size, max_len, max_len
        return decoder_outputs.transpose(0, 1)