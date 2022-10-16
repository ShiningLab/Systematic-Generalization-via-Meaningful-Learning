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
# private
from .encoder import PositionalEncoding


class ModelGraph(nn.Module):
    """docstring for ModelGraph""" 
    def __init__(self, config):
        super(ModelGraph, self).__init__()
        self.config = config
        # embedding layers
        self.src_embedding_layer = nn.Embedding(config.src_vocab_size, config.embedding_size)
        self.tgt_embedding_layer = nn.Embedding(config.tgt_vocab_size, config.embedding_size)
        # positional encoder
        self.pos_encoder = PositionalEncoding(config.embedding_size, config.pos_encoder_drop_rate)
        # transformer model
        self.transformer_model = nn.Transformer(d_model=config.en_hidden_size,
                                                nhead=config.num_heads,
                                                num_encoder_layers=config.tfm_en_num_layers,
                                                num_decoder_layers=config.tfm_de_num_layers,
                                                dim_feedforward=config.ffnn_dim,
                                                dropout=config.en_drop_rate)
        # generator layer (a.k.a. the final linear layer)
        # encoder hidden dim = decoder hidden dim
        self.generator = nn.Linear(config.en_hidden_size, config.tgt_vocab_size)  

    def forward(self, xs, x_lens, ys):
        # xs: batch_size, src_seq_len
        # x_lens: batch_size
        # ys: batch_size, max_ys_seq_len
        # batch_size, src_seq_len
        max_ys_seq_len = ys.shape[1]
        src_key_padding_mask = (xs == self.config.PAD_IDX)
        # src_seq_len, batch_size, emb_size
        xs = (self.src_embedding_layer(xs) * math.sqrt(self.config.embedding_size)).transpose(0, 1)
        xs = self.pos_encoder(xs)
        if self.training: 
            # batch_size, 1
            start_idxes = torch.ones([ys.size(0), 1], 
                dtype=torch.int64, device=self.config.device)*self.config.BOS_IDX
            # batch_size, tgt_seq_len
            # right shift decoder input by adding start symbol
            ys = torch.cat((start_idxes, ys), 1)[:, :-1] 
            # tgt_seq_len, batch_size, emb_size
            ys = (self.tgt_embedding_layer(ys) * math.sqrt(self.config.embedding_size)).transpose(0, 1)
            ys = self.pos_encoder(ys)
            # tgt_seq_len, tgt_seq_len
            tgt_mask = self.transformer_model.generate_square_subsequent_mask(ys.size(0)).to(self.config.device)
            # tgt_seq_len, batch_size, de_hidden_size
            decoder_outputs = self.transformer_model(
                src=xs, 
                tgt=ys, 
                tgt_mask=tgt_mask, 
                src_key_padding_mask=src_key_padding_mask)
            # tgt_seq_len, batch_size, tgt_vocab_size
            decoder_outputs = F.log_softmax(self.generator(decoder_outputs), dim=-1)
        else:
            # src_seq_len, batch_size, en_hidden_size
            encoder_hidden_states = self.transformer_model.encoder(
                src=xs, 
                src_key_padding_mask=src_key_padding_mask)
            # batch_size, tgt_seq_len+1
            decoder_inputs = torch.empty(
                (xs.size(1), max_ys_seq_len+1), 
                dtype=torch.int64, device=self.config.device).fill_(self.config.BOS_IDX)
            # max_ys_seq_len, batch_size, vocab_size
            decoder_outputs = torch.zeros(
                max_ys_seq_len, 
                xs.size(1), 
                self.config.tgt_vocab_size, 
                device=self.config.device)
            for i in range(max_ys_seq_len): 
                # cur_seq_len, batch_size, emb_size
                decoder_input = (self.tgt_embedding_layer(decoder_inputs[:, :i+1]) * math.sqrt(self.config.embedding_size)).transpose(0, 1)
                decoder_input = self.pos_encoder(decoder_input)
                # cur_seq_len, cur_seq_len
                tgt_mask = self.transformer_model.generate_square_subsequent_mask(i+1).to(self.config.device)
                # cur_seq_len, batch_size, de_hidden_size
                decoder_output = self.transformer_model.decoder(
                    tgt=decoder_input, 
                    memory=encoder_hidden_states, 
                    tgt_mask=tgt_mask, 
                    memory_key_padding_mask=src_key_padding_mask)
                # batch_size, tgt_vocab_size
                decoder_output = self.generator(decoder_output)[-1]
                # batch_size, tgt_vocab_size
                decoder_outputs[i] = decoder_output
                # batch_size, cur_seq_len
                decoder_inputs[:, i+1] = decoder_output.max(1)[1]
            decoder_outputs = F.log_softmax(decoder_outputs, dim=-1)
        # batch_size, tgt_seq_len
        return decoder_outputs.transpose(0, 1)