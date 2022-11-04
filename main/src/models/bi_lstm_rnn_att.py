#!/usr/bin/env python
# -*- coding:utf-8 -*-
__author__ = 'Shining'
__email__ = 'ning.shi@ualberta.ca'


# built-in
import random
# public
import torch
import torch.nn as nn
# private
from .encoder import BiLSTMRNNEncoder
from .decoder import AttBiLSTMRNNDecoder


class ModelGraph(nn.Module): 
    """docstring for ModelGraph""" 
    def __init__(self, config): 
        super(ModelGraph, self).__init__() 
        self.config = config
        self.encoder = BiLSTMRNNEncoder(config)
        self.decoder = AttBiLSTMRNNDecoder(config)

    def forward(self, xs, x_lens, ys, teacher_forcing_ratio=0.5):
        # xs: batch_size, max_xs_seq_len
        # x_lens: batch_size
        # ys: batch_size, max_ys_seq_len
        batch_size = xs.shape[0]
        max_ys_seq_len = ys.shape[1]
        teacher_forcing_ratio = self.config.teacher_forcing_ratio if self.training else 0.
        # encoder_output: batch_size, max_xs_seq_len, en_hidden_size
        # decoder_hidden: (h, c)
        # h, c: 1, batch_size, en_hidden_size
        encoder_output, decoder_hidden = self.encoder(xs, x_lens)
        # batch_size
        decoder_input = torch.empty(
            batch_size, 
            dtype=torch.int64, 
            device=self.config.device)
        decoder_input.fill_(self.config.BOS_IDX)
        # max_ys_seq_len, batch_size, vocab_size
        decoder_outputs = torch.zeros(
            max_ys_seq_len, 
            batch_size, 
            self.config.tgt_vocab_size, 
            device=self.config.device)
        # greedy search with teacher forcing
        for i in range(max_ys_seq_len):
            # decoder_output: batch_size, vocab_size
            # decoder_hidden: (h, c)
            # h, c: 1, batch_size, en_hidden_size
            decoder_output, decoder_hidden = self.decoder(
                decoder_input, decoder_hidden, encoder_output, x_lens)
            # batch_size, vocab_size
            decoder_outputs[i] = decoder_output
            # batch_size
            decoder_input = ys[:, i] if random.random() < teacher_forcing_ratio \
            else decoder_output.max(1)[1]
        # batch_size, max_ys_seq_len, vocab_size
        return decoder_outputs.transpose(0, 1)