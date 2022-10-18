#!/usr/bin/env python
# -*- coding:utf-8 -*-
__author__ = 'Shining'
__email__ = 'mrshininnnnn@gmail.com'


# public
import torch
import torch.nn as nn
import torch.nn.functional as F
# private
from .attention import LSTMRNNDecoderAttention


class AttBiLSTMRNNDecoder(nn.Module):
    """Bidirectional RNN Decoder with Long Short Term Memory (LSTM) Unit and Attention"""
    def __init__(self, config):
        super(AttBiLSTMRNNDecoder, self).__init__()
        self.config = config
        self.embedding = nn.Embedding(
            num_embeddings=self.config.tgt_vocab_size, 
            embedding_dim=self.config.embedding_size, 
            padding_idx=self.config.PAD_IDX)
        self.em_dropout=nn.Dropout(self.config.embedding_drop_rate)
        self.attn = LSTMRNNDecoderAttention(self.config)
        self.attn_combine = torch.nn.Linear(
            self.config.embedding_size + self.config.en_hidden_size, 
            self.config.de_hidden_size)
        self.lstm = nn.LSTM(
            input_size=self.config.embedding_size, 
            hidden_size=self.config.de_hidden_size, 
            num_layers=self.config.de_num_layers, 
            batch_first=True, 
            dropout=0, 
            bidirectional=False)
        self.lstm_dropout = nn.Dropout(self.config.de_drop_rate)
        self.out = torch.nn.Linear(self.config.de_hidden_size, self.config.tgt_vocab_size)
        self.softmax = torch.nn.LogSoftmax(dim=1)

    def forward(self, x, hidden, encoder_output, src_lens):
        # x: batch_size
        # hidden: (h, c)
        # h, c: 1, batch_size, de_hidden_size
        # encoder_output: batch_size, max_src_seq_len, en_hidden_size
        # src_lens: batch_size
        # batch_size, 1, embedding_dim
        x = self.embedding(x).unsqueeze(1)
        x = self.em_dropout(x)
        # batch_size, 1, max_src_seq_len
        attn_w = self.attn(hidden, encoder_output, src_lens)
        # batch_size, 1, en_hidden_size
        context = attn_w.bmm(encoder_output)
        # batch_size, 1, de_hidden_size
        x = self.attn_combine(torch.cat((x, context), 2))
        x = F.relu(x)
        x, (h, c) = self.lstm(x, hidden)
        # batch_size, de_hidden_size
        x = self.lstm_dropout(x).squeeze(1)
        # batch_size, tgt_vocab_size
        x = self.out(x)
        # batch_size, vocab_size
        x = self.softmax(x)
        return x, (h, c)


# adapted from https://github.com/bentrevett/pytorch-seq2seq/
class CNNDecoder(nn.Module):
    def __init__(self, 
                 output_dim, 
                 emb_dim, 
                 hid_dim, 
                 n_layers, 
                 kernel_size, 
                 dropout, 
                 trg_pad_idx, 
                 device,
                 max_length=100):
        super().__init__()
        
        self.kernel_size = kernel_size
        self.trg_pad_idx = trg_pad_idx
        self.device = device
        
        self.scale = torch.sqrt(torch.FloatTensor([0.5])).to(device)
        
        self.tok_embedding = nn.Embedding(output_dim, emb_dim)
        self.pos_embedding = nn.Embedding(max_length, emb_dim)
        
        self.emb2hid = nn.Linear(emb_dim, hid_dim)
        self.hid2emb = nn.Linear(hid_dim, emb_dim)
        
        self.attn_hid2emb = nn.Linear(hid_dim, emb_dim)
        self.attn_emb2hid = nn.Linear(emb_dim, hid_dim)
        
        self.fc_out = nn.Linear(emb_dim, output_dim)
        
        self.softmax = torch.nn.LogSoftmax(dim=-1)

        self.convs = nn.ModuleList([nn.Conv1d(in_channels = hid_dim, 
                                              out_channels = 2 * hid_dim, 
                                              kernel_size = kernel_size)
                                    for _ in range(n_layers)])
        
        self.dropout = nn.Dropout(dropout)
      
    def calculate_attention(self, embedded, conved, encoder_conved, encoder_combined):
        # embedded = [batch size, trg len, emb dim]
        # conved = [batch size, hid dim, trg len]
        # encoder_conved = encoder_combined = [batch size, src len, emb dim]
        
        # permute and convert back to emb dim
        # conved_emb = [batch size, trg len, emb dim]
        conved_emb = self.attn_hid2emb(conved.permute(0, 2, 1))
        
        # combined = [batch size, trg len, emb dim]
        combined = (conved_emb + embedded) * self.scale
        # energy = [batch size, trg len, src len]
        energy = torch.matmul(combined, encoder_conved.permute(0, 2, 1))
        # attention = [batch size, trg len, src len]
        attention = F.softmax(energy, dim=2)
        # attended_encoding = [batch size, trg len, emd dim]
        attended_encoding = torch.matmul(attention, encoder_combined)
        # convert from emb dim -> hid dim
        # attended_encoding = [batch size, trg len, hid dim]
        attended_encoding = self.attn_emb2hid(attended_encoding)
        # apply residual connection
        # attended_combined = [batch size, hid dim, trg len]
        attended_combined = (conved + attended_encoding.permute(0, 2, 1)) * self.scale
        return attention, attended_combined
        
    def forward(self, trg, encoder_conved, encoder_combined):
        # trg = [batch size, trg len]
        # encoder_conved = encoder_combined = [batch size, src len, emb dim]
        batch_size = trg.shape[0]
        trg_len = trg.shape[1]
        # create position tensor
        # pos = [batch size, trg len]
        pos = torch.arange(0, trg_len).unsqueeze(0).repeat(batch_size, 1).to(self.device)
        # embed tokens and positions
        # tok_embedded = [batch size, trg len, emb dim]
        # pos_embedded = [batch size, trg len, emb dim]
        tok_embedded = self.tok_embedding(trg)
        pos_embedded = self.pos_embedding(pos)
        # combine embeddings by elementwise summing
        # embedded = [batch size, trg len, emb dim]
        embedded = self.dropout(tok_embedded + pos_embedded)
        # pass embedded through linear layer to go through emb dim -> hid dim
        # conv_input = [batch size, trg len, hid dim]
        conv_input = self.emb2hid(embedded)
        # permute for convolutional layer
        # conv_input = [batch size, hid dim, trg len]
        conv_input = conv_input.permute(0, 2, 1) 
        batch_size = conv_input.shape[0]
        hid_dim = conv_input.shape[1]
        for i, conv in enumerate(self.convs):
            # apply dropout
            conv_input = self.dropout(conv_input)
            # need to pad so decoder can't "cheat"
            padding = torch.zeros(
                batch_size, 
                hid_dim, 
                self.kernel_size - 1).fill_(self.trg_pad_idx).to(self.device)
            # padded_conv_input = [batch size, hid dim, trg len + kernel size - 1]
            padded_conv_input = torch.cat((padding, conv_input), dim = 2)
            # pass through convolutional layer
            # conved = [batch size, 2 * hid dim, trg len]
            conved = conv(padded_conv_input)
            # pass through GLU activation function
            conved = F.glu(conved, dim = 1)
            # calculate attention
            # attention = [batch size, trg len, src len]
            attention, conved = self.calculate_attention(
                embedded, 
                conved, 
                encoder_conved, 
                encoder_combined)
            # apply residual connection
            # conved = [batch size, hid dim, trg len]
            conved = (conved + conv_input) * self.scale
            # set conv_input to conved for next loop iteration
            conv_input = conved
        # conved = [batch size, trg len, emb dim]
        conved = self.hid2emb(conved.permute(0, 2, 1))
        # output = [batch size, trg len, output dim]
        output = self.softmax(self.fc_out(self.dropout(conved)))
        return output, attention 