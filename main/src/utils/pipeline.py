#!/usr/bin/env python
# -*- coding:utf-8 -*-
__author__ = 'Shining'
__email__ = 'ning.shi@ualberta.ca'


# public
import torch
from torch.utils import data as torch_data
# private
from ..models import bi_lstm_rnn_att, cnn_att, transformer


class Dataset(torch_data.Dataset):
    """docstring for Dataset"""
    def __init__(self, data_dict):
        super(Dataset, self).__init__()
        self.xs = data_dict['xs']
        self.ys = data_dict['ys']
        self.data_size = len(self.xs)

    def __len__(self):
        return self.data_size

    def __getitem__(self, idx):
        return self.xs[idx], self.ys[idx]

def pick_model(config):
    """
    To pick the model structure given configurations
    """
    if config.mode == 'seq2seq':
        # bi-directional LSTM with attention
        if config.model_name == 'bi_lstm_rnn_att':
            return bi_lstm_rnn_att.ModelGraph(config).to(config.device)
        # CNN with attention
        elif config.model_name == 'cnn_att':
            return cnn_att.ModelGraph(config).to(config.device)
        # a
        elif config.model_name == 'transformer':
            return transformer.ModelGraph(config).to(config.device)
    raise NotImplementedError

def init_parameters(model): 
    """
    To initialize model parameters
    """
    for name, parameters in model.named_parameters(): 
        if 'weight' in name: 
            torch.nn.init.normal_(parameters.data, mean=0, std=0.01)
        else:
            torch.nn.init.constant_(parameters.data, 0)

def count_parameters(model): 
    # get total size of trainable parameters 
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def show_config(config, model):
    # general information
    general_info = '\n*Configuration*'
    general_info += '\nmodel name: {}'.format(config.model_name)
    general_info += '\ntrainable parameters:{:,.0f}'.format(config.num_parameters)
    model_info = '\nmodel structure:'
    for parameters in model.state_dict():
        model_info += '\n{}\t{}'.format(parameters, model.state_dict()[parameters].size())
    general_info += model_info
    general_info += '\ndevice: {}'.format(config.device)
    general_info += '\nuse gpu: {}'.format(config.use_gpu)
    general_info += '\ntrain size: {}'.format(config.train_size)
    general_info += '\ntest size: {}'.format(config.test_size)
    general_info += '\nsource vocab size: {}'.format(config.src_vocab_size)
    general_info += '\ntarget vocab size: {}'.format(config.tgt_vocab_size)
    general_info += '\nbatch size: {}'.format(config.batch_size)
    general_info += '\ntrain batch: {}'.format(config.train_batch)
    general_info += '\ntest batch: {}'.format(config.test_batch)
    general_info += '\nif load check point: {}'.format(config.load_check_point)
    if config.load_check_point:
        general_info += '\nModel restored from {}'.format(config.SAVE_POINT)
    else:
        general_info += '\nModel saved to {}'.format(config.SAVE_POINT)
    general_info += '\n'
    print(general_info)

    return general_info

def translate(seq: list, trans_dict: dict) -> list: 
    return [trans_dict[token] for token in seq]

def post_process(idx_ys_, tgt_idx2vocab_dict, config):
    vocab_ys_ = []
    for i in range(len(idx_ys_)):
        y_ = idx_ys_[i].tolist()
        if config.EOS_IDX in y_:
            y_ = y_[:y_.index(config.EOS_IDX)]
        vocab_ys_.append(y_)
    vocab_ys_ = [translate(y_, tgt_idx2vocab_dict) for y_ in vocab_ys_]
    return vocab_ys_

def save_model(step, epoch, model_state_dict, opt_state_dict, path):
    # save model, optimizer, and everything required to keep
    checkpoint_to_save = {
        'step': step, 
        'epoch': epoch, 
        'model': model_state_dict(), 
        'optimizer': opt_state_dict()}
    torch.save(checkpoint_to_save, path)
    print('Model saved as {}.'.format(path))  

def preprocess(xs, ys, src_vocab2idx_dict, tgt_vocab2idx_dict, config): 
    xs = [translate(x, src_vocab2idx_dict) for x in xs]
    ys = [translate(y, tgt_vocab2idx_dict) for y in ys]
    xs = [torch.Tensor(x) for x in xs]
    ys = [torch.Tensor(y + [config.EOS_IDX]) for y in ys] 
    return xs, ys

def padding(seqs):
    seq_lens = [len(seq) for seq in seqs]
    max_len = max(seq_lens)
    # default pad index is 0
    padded_seqs = torch.zeros([len(seqs), max_len]).long()
    for i, seq in enumerate(seqs): 
        seq_len = seq_lens[i]
        padded_seqs[i, :seq_len] = seq[:seq_len]
    return padded_seqs, torch.Tensor(seq_lens)