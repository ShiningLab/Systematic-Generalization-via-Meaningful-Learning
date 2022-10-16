#!/usr/bin/env python
# -*- coding:utf-8 -*-
__author__ = 'Shining'
__email__ = 'mrshininnnnn@gmail.com'


# public
import torch
import torch.nn as nn
# private
from .encoder import CNNEncoder
from .decoder import CNNDecoder


class ModelGraph(nn.Module):
    """docstring for ModelGraph"""
    def __init__(self, config): 
        super(ModelGraph, self).__init__() 
        self.config = config
        self.encoder = CNNEncoder(
            input_dim=config.src_vocab_size
            , emb_dim=config.embedding_size
            , hid_dim=config.en_hidden_size
            , n_layers=config.cnn_en_num_layers
            , kernel_size=config.kernel_size
            , dropout=config.en_drop_rate
            , device=config.device
            , max_length=config.cnn_en_max_length)
        self.decoder = CNNDecoder(
            output_dim=config.tgt_vocab_size
            , emb_dim=config.embedding_size
            , hid_dim=config.de_hidden_size
            , n_layers=config.cnn_de_num_layers
            , kernel_size=config.kernel_size
            , dropout=config.de_drop_rate
            , trg_pad_idx=config.PAD_IDX
            , device=config.device
            , max_length=config.cnn_de_max_length)

    def forward(self, xs, x_lens, ys):
        # xs: batch_size, src_seq_len
        # x_lens: batch_size
        # ys: batch_size, max_ys_seq_len
        batch_size = xs.shape[0]
        max_ys_seq_len = ys.shape[1]
        # batch_size, src_seq_len, en_hidden_size
        encoder_conved, encoder_combined = self.encoder(xs)
        if self.training: 
            start_idxes = torch.ones([batch_size, 1], 
                dtype=torch.int64, device=self.config.device)*self.config.BOS_IDX
            # right shift decoder input by adding start symbol
            ys = torch.cat((start_idxes, ys), 1)[:, :-1] 
            # batch_size, tgt_seq_len, tgt_vocab_size
            decoder_outputs, _ = self.decoder(ys, encoder_conved, encoder_combined)
            return decoder_outputs
        else:
            # batch_size, tgt_seq_len+1
            decoder_inputs = torch.empty(
                (batch_size, max_ys_seq_len+1), 
                dtype=torch.int64, device=self.config.device).fill_(self.config.BOS_IDX)
            # max_ys_seq_len, batch_size, vocab_size
            decoder_outputs = torch.zeros(
                max_ys_seq_len, 
                batch_size, 
                self.config.tgt_vocab_size, 
                device=self.config.device)
            for i in range(max_ys_seq_len):
                # batch_size, cur_seq_len, tgt_vocab_size
                decoder_output, _ = self.decoder(decoder_inputs[:, :i+1], encoder_conved, encoder_combined)
                # batch_size, tgt_vocab_size
                decoder_output = decoder_output.transpose(0, 1)[-1]
                # batch_size, vocab_size
                decoder_outputs[i] = decoder_output
                # batch_size
                decoder_inputs[:, i+1] = decoder_output.max(1)[1]
            # batch_size, tgt_seq_len, vocab_size
            return decoder_outputs.transpose(0, 1)
