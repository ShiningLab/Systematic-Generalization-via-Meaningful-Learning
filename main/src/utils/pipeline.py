#!/usr/bin/env python
# -*- coding:utf-8 -*-
__author__ = 'Author'
__email__ = 'Email'


import torch
# torch.manual_seed(0)
from torch.utils import data as torch_data

# import copy
# import random
# import numpy as np
# np.random.seed(0)
# import Levenshtein 

# # private
# from ..models import (
#     transformer, 
#     gru_rnn, lstm_rnn, 
#     bi_gru_rnn, bi_lstm_rnn, 
#     bi_gru_rnn_att, bi_lstm_rnn_att)
from ..models import bi_lstm_rnn_att

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
        

# class OfflineDataset(torch_data.Dataset):
#     """docstring for OfflineDataset"""
#     def __init__(self, data_dict):
#         super(OfflineDataset, self).__init__()
#         self.xs = data_dict['xs']
#         self.ys = data_dict['ys']
#         if 'ys_' in data_dict:
#             self.ys_ = data_dict['ys_']
#         else:
#             self.ys_ = None
#         self.data_size = len(self.ys)
        
#     def __len__(self): 
#         return self.data_size

#     def __getitem__(self, idx): 
#         if self.ys_ is None: 
#             return self.xs[idx], self.ys[idx] 
#         else:
#             return self.xs[idx], self.ys_[idx]


# class OnlineDataset(torch_data.Dataset):
#     """docstring for OnlineDataset"""
#     def __init__(self, data_dict):
#         super(OnlineDataset, self).__init__() 
#         self.ys = data_dict['ys']
#         self.data_size = len(self.ys)

#     def __len__(self): 
#         return self.data_size

#     def __getitem__(self, idx): 
#         return self.ys[idx]


def pick_model(config):
    # auto regressive
    if config.auto_regressive:
        if config.model_name == 'bi_lstm_rnn_att':
            return bi_lstm_rnn_att.ModelGraph(config).to(config.device)

#     if config.model_name == 'transformer':
#         if method == 'e2e':
#             return transformer.E2EModelGraph(config).to(config.device)
#         elif method == 'rec':
#             return transformer.RecModelGraph(config).to(config.device)

#     elif config.model_name == 'gru_rnn':
#         if method == 'e2e':
#             return gru_rnn.E2EModelGraph(config).to(config.device)
#         elif method == 'rec':
#             return gru_rnn.RecModelGraph(config).to(config.device)

#     elif config.model_name == 'lstm_rnn':
#         if method == 'e2e':
#             return lstm_rnn.E2EModelGraph(config).to(config.device)
#         elif method == 'rec':
#             return lstm_rnn.RecModelGraph(config).to(config.device)

#     elif config.model_name == 'bi_gru_rnn':
#         if method == 'e2e':
#             return bi_gru_rnn.E2EModelGraph(config).to(config.device)
#         elif method == 'rec':
#             return bi_gru_rnn.RecModelGraph(config).to(config.device)

#     elif config.model_name == 'bi_lstm_rnn':
#         if method == 'e2e':
#             return bi_lstm_rnn.E2EModelGraph(config).to(config.device)
#         elif method == 'rec':
#             return bi_lstm_rnn.RecModelGraph(config).to(config.device)

#     elif config.model_name =='bi_gru_rnn_att':
#         if method == 'e2e':
#             return bi_gru_rnn_att.E2EModelGraph(config).to(config.device)
#         elif method == 'rec':
#             return bi_gru_rnn_att.RecModelGraph(config).to(config.device)

#     elif config.model_name =='bi_lstm_rnn_att':
#         if method == 'e2e':
#             return bi_lstm_rnn_att.E2EModelGraph(config).to(config.device)
#         elif method == 'rec':
#             return bi_lstm_rnn_att.RecModelGraph(config).to(config.device)

#     else:
#         raise ValueError('Wrong model to pick.')

# def get_list_mean(l: list) -> float:
#     return sum(l) / len(l)

def init_parameters(model): 
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
    general_info += '\nmodel: {}'.format(config.model_name)
    general_info += '\ntrainable parameters:{:,.0f}'.format(config.num_parameters)
    model_info = '\nmodel:'
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
    general_info += '\n'
    print(general_info)

    return general_info

def translate(seq: list, trans_dict: dict) -> list: 
    return [trans_dict[token] for token in seq]

# def rm_idx(seq, idx):
#     return [i for i in seq if i != idx]

def post_process(ys_, raw_data, tgt_idx2vocab_dict):
    raw_xs, raw_ys = raw_data
    ys_ = [y_[:len(y)] for y, y_ in zip(raw_ys, ys_)]
    ys_ = [translate(y_, tgt_idx2vocab_dict) for y_ in ys_]
    return raw_xs, raw_ys, ys_

def save_model(step, epoch, model_state_dict, opt_state_dict, path):
    # save model, optimizer, and everything required to keep
    checkpoint_to_save = {
        'step': step, 
        'epoch': epoch, 
        'model': model_state_dict(), 
        'optimizer': opt_state_dict()}
    torch.save(checkpoint_to_save, path)
    print('Model saved as {}.'.format(path))  

# def save_check_point(step, epoch, model_state_dict, opt_state_dict, path):
#     # save model, optimizer, and everything required to keep
#     checkpoint_to_save = {
#         'step': step, 
#         'epoch': epoch, 
#         'model': model_state_dict(), 
#         'optimizer': opt_state_dict()}
#     torch.save(checkpoint_to_save, path)
#     print('Model saved as {}.'.format(path))

# def rand_sample(srcs, tars, preds, src_dict, tar_dict, pred_dict): 
#     src, tar, pred = random.choice([(src, tar, pred) for src, tar, pred in zip(srcs, tars, preds)])
#     src = translate(src, src_dict)
#     tar = translate(tar, tar_dict)
#     pred = translate(pred, pred_dict)
#     return ' '.join(src), ' '.join(tar), ' '.join(pred)

# # for swap sort
# def find_src_index_to_swap(x: list, y: list) -> int:
#     if x == y:
#         return -1
#     else:
#         idx_to_swap = [i for i in range(len(x)) if x[i] != y[i]][0]
#         return idx_to_swap
    
# def find_tgt_index_to_swap(x: list, src_idx: int) -> int:
#     if src_idx == -1:
#         return -1
#     else:
#         return np.argmin(x[src_idx:]) + src_idx

# def convert_to_int(seq: list) -> list:
#     return [int(str_number) for str_number in seq]

# def convert_to_str(seq: list) -> str:
#     return [str(int_number) for int_number in seq]

# # class for data generation of the Arithmetic Equation Simplification (AES) problem 
# class ArithmeticEquationSimplification(): 
#     """docstring for ArithmeticEquationSimplification"""
#     def __init__(self, config):
#         super().__init__()
#         self.operators = config.operators
#         self.pos_digits = np.arange(2, config.N+2).tolist()
#         self.neg_digits = np.arange(-config.N, -1).tolist()
#         self.digits = self.pos_digits + self.neg_digits
#         self.base_dict = self.gen_base_dict()

#     def gen_base_dict(self):
#         base_dict = {str(i):[] for i in self.pos_digits}
#         for a in self.digits:
#             for o in self.operators:
#                 for b in self.pos_digits:
#                     try:
#                         e = [str(a), o, str(b)]
#                         v = str(eval(''.join(e)))
#                         e[0] = e[0].replace('-', '- ')
#                         e = ' '.join(list(e))
#                         if v in base_dict: 
#                             base_dict[v].append('( {} )'.format(e))
#                     except:
#                         pass
#         return base_dict

#     def replace_numbers(self, ys):
#         ys = copy.deepcopy(ys)
#         xs = []
#         for y in ys: 
#             num_idx = [i for i, token in enumerate(y) if token.isdigit()]
#             num_to_replace = np.random.choice(range(len(num_idx)+1))
#             idx_to_replace = np.random.choice(num_idx, num_to_replace, False)
#             for i in idx_to_replace:
#                 y[i] = np.random.choice(self.base_dict[y[i]])
#             xs.append(' '.join(y).split())
#         return xs


# # class for data generation of the Arithmetic Equation Correction (AEC) problem 
# class ArithmeticEquationCorrection(): 
#     """docstring for ArithmeticEquationCorrection"""
#     def __init__(self, config):
#         super().__init__()
#         self.operators = config.operators
#         self.num_errors = config.num_errors
#         self.pos_digits = np.arange(2, config.N+2).tolist()
#         self.neg_digits = np.arange(-config.N, -1).tolist()
#         self.digits = self.pos_digits + self.neg_digits
        
#         def delete(tk_y, idx): 
#             tk_y[idx] = ''
#             return tk_y
#         def insert(tk_y, idx): 
#             tk_y[idx] = str(np.random.choice(self.operators+self.pos_digits)) + ' ' + tk_y[idx]
#             return tk_y 
#         def sub(tk_y, idx):
#             tk_y[idx] = str(np.random.choice(self.operators+self.pos_digits))
#             return tk_y
        
#         self.trans_funs = [delete, insert, sub]
    
#     def transform(self, y, idxes): 
#         tk_y = y.copy()
#         for idx in idxes: 
#             f = np.random.choice(self.trans_funs)
#             tk_y = f(tk_y, idx)
#         return tk_y
        
#     def random_transform(self, ys): 
#         xs = []
#         for y in ys:
#             y_len = len(y) - 1
#             num_idxes = np.random.choice(range(self.num_errors+1))
#             idxes = sorted(np.random.choice(range(y_len), num_idxes, False))
#             x = self.transform(y, idxes)
#             xs.append(' '.join([i for i in x if len(i)>0]).split())
#         return xs


# def levenshtein_editops_list(source, target):
#     unique_elements = sorted(set(source + target)) 
#     char_list = [chr(i) for i in range(len(unique_elements))]
#     if len(unique_elements) > len(char_list):
#         raise Exception("too many elements")
#     else:
#         unique_element_map = {ele:char_list[i]  for i, ele in enumerate(unique_elements)}
#     source_str = ''.join([unique_element_map[ele] for ele in source])
#     target_str = ''.join([unique_element_map[ele] for ele in target])
#     transform_list = Levenshtein.editops(source_str, target_str)
#     return transform_list

# def aes_sampler(ys: list, aes) -> list: 
#     xs = aes.replace_numbers(ys.copy())
#     return [(x, y) for x, y in zip(xs, ys)]

# def aec_sampler(ys: list, aec) -> list:
#     xs = aec.random_transform(ys.copy())
#     return [(x, y) for x, y in zip(xs, ys)]

# def inverse_sampler(data, data_src, aes=None, aec=None): 
#     if data_src == 'aes': 
#         return aes_sampler(data, aes) 
#     elif data_src == 'aor': 
#         return data
#     elif data_src == 'aec': 
#         return aec_sampler(data, aec)

# def e2e_online_generator(data_src: str, data) -> list:
#     # online training data generation for end2end
#     # for Arithmetic Equation Simplification (AES) 
#     if data_src == 'aes': 
#         x, y = data
#         xs = [x.copy()]
#         num_left = len([i for i in x if i == '('])
#         for i in range(num_left):
#             left_idx = x.index('(') 
#             right_idx = x.index(')') 
#             v = y[left_idx] 
#             x = x[:left_idx] + [v] + x[right_idx+1:]
#             xs.append(x)
#         index = np.random.choice(range(len(xs)))
#         x = xs[index]
#         return x, y
#     # for Arithmetic Operators Restoration (AOR)
#     elif data_src == 'aor':
#         # make a copy
#         y = data.copy()
#         x = y.copy()
#         # get operator indexes
#         operator_idxes = [i for i, token in enumerate(y) if not token.isdigit()][::-1]
#         # decide how many operators to remove
#         num_idxes = np.random.choice(range(len(operator_idxes)+1))
#         # decide operators to remove
#         idxes_to_remove = operator_idxes[:num_idxes]
#         x = [x[i] for i in range(len(x)) if i not in idxes_to_remove]
#         return x, y
#     # for Arithmetic Equation Correction (AEC)
#     elif data_src == 'aec': 
#         # make a copy
#         x, y = data
#         xs = [x.copy()] 
#         editops = levenshtein_editops_list(x, y)
#         if len(editops) == 0: 
#             return x, y
#         else:
#             c = 0 
#             for tag, i, j in editops: 
#                 i += c
#                 if tag == 'replace':
#                     x[i] = y[j]
#                 elif tag == 'delete':
#                     del x[i]
#                     c -= 1
#                 elif tag == 'insert':
#                     x.insert(i, y[j]) 
#                     c += 1
#                 xs.append(x.copy())
#             index = np.random.choice(range(len(xs)))
#             x = xs[index]
#         return x, y

# def rec_online_generator(data_src: str, data: list) -> list:
#     # online training data generation for recurrent inference
#     # for Arithmetic Equation Simplification (AES) 
#     if data_src == 'aes': 
#         x, y = data
#         xs = [x.copy()]
#         ys_ = []
#         num_left = len([i for i in x if i == '('])
#         for i in range(num_left):
#             left_idx = x.index('(') 
#             right_idx = x.index(')') 
#             v = y[left_idx] 
#             ys_.append(['<pos_{}>'.format(left_idx), '<pos_{}>'.format(right_idx), v])
#             x = x[:left_idx] + [v] + x[right_idx+1:]
#             xs.append(x)
#         ys_.append(['<done>']*3)
#         index = np.random.choice(range(len(xs)))
#         x = xs[index]
#         y_ = ys_[index]
#         return x, y_
#     # for Arithmetic Operators Restoration (AOR)
#     elif data_src == 'aor':
#         # make a copy
#         y = data.copy() # list
#         x = y.copy()
#         # get operator indexes
#         operator_idxes = [i for i, token in enumerate(y) if not token.isdigit()][::-1]
#         # decide how many operators to remove
#         num_idxes = np.random.choice(range(len(operator_idxes)+1))
#         if num_idxes == 0:
#             return x, ['<done>', '<done>']
#         else:
#             # decide operators to remove
#             idxes_to_remove = operator_idxes[:num_idxes]
#             # generat label
#             y_ = ['<pos_{}>'.format(idxes_to_remove[-1]), x[idxes_to_remove[-1]]]
#             # generate sample
#             x = [x[i] for i in range(len(x)) if i not in idxes_to_remove]
#             return x, y_
#     # for Arithmetic Equation Correction (AEC)
#     elif data_src == 'aec': 
#         x, y = data
#         xs = [x.copy()]
#         ys_ = []
#         editops = levenshtein_editops_list(x, y)
#         if len(editops) == 0: 
#             y_ = ['<done>']*3 
#             return x, y_ 
#         else:
#             c = 0 
#             for tag, i, j in editops: 
#                 i += c
#                 if tag == 'replace':
#                     y_ = ['<sub>', '<pos_{}>'.format(i), y[j]]
#                     x[i] = y[j]
#                 elif tag == 'delete': 
#                     # y_ = ['<delete>', '<pos_{}>'.format(i), '<done>'] 
#                     y_ = ['<delete>', '<pos_{}>'.format(i), '<pos_{}>'.format(i)]
#                     del x[i]
#                     c -= 1
#                 elif tag == 'insert': 
#                     y_ = ['<insert>', '<pos_{}>'.format(i), y[j]]
#                     x.insert(i, y[j]) 
#                     c += 1
#                 xs.append(x.copy()) 
#                 ys_.append(y_)
#             ys_.append(['<done>']*3)
#             index = np.random.choice(range(len(xs)))
#             x = xs[index]
#             y_ = ys_[index]
#         return x, y_

# def rec_offline_generator(data_src: str, data) -> list: 
#     # for Arithmetic Equation Simplification (AES) 
#     if data_src == 'aes':
#         x, y = data
#         num_left = len([i for i in x if i == '('])
#         if num_left == 0:
#             y_ = ['<done>']*3
#         else:
#             left_idx = x.index('(') 
#             right_idx = x.index(')') 
#             v = y[left_idx] 
#             y_ = ['<pos_{}>'.format(left_idx), '<pos_{}>'.format(right_idx), v]
#         return x, y_
#     # for Arithmetic Operators Restoration (AOR)
#     elif data_src == 'aor': 
#         return data
#     # for Arithmetic Equation Correction (AEC)
#     elif data_src == 'aec': 
#         x, y = data
#         editops = levenshtein_editops_list(x, y)
#         if len(editops) == 0:
#             y_ = ['<done>']*3
#         else:
#             tag, i, j = editops[0]
#             if tag == 'replace':
#                 y_ = ['<sub>', '<pos_{}>'.format(i), y[j]]
#             elif tag == 'delete': 
#                 # y_ = ['<delete>', '<pos_{}>'.format(i), '<done>'] 
#                 y_ = ['<delete>', '<pos_{}>'.format(i), '<pos_{}>'.format(i)] 
#             elif tag == 'insert': 
#                 y_ = ['<insert>', '<pos_{}>'.format(i), y[j]] 
#         return x, y_

# def tag_online_generator(data_src: str, data) -> list:
#     # for Arithmetic Equation Simplification (AES) 
#     if data_src == 'aes': 
#         # pick an intermediate step
#         x, y = data
#         xs = [x.copy()]
#         num_left = len([i for i in x if i == '('])
#         for i in range(num_left):
#             left_idx = x.index('(') 
#             right_idx = x.index(')') 
#             v = y[left_idx] 
#             x = x[:left_idx] + [v] + x[right_idx+1:]
#             xs.append(x)
#         index = np.random.choice(range(len(xs)))
#         x = xs[index]
#         # convert to a tagging sequence
#         y_ = []
#         x_ = x.copy()
#         x_token = x_.pop(0)
#         for i in range(len(y)):
#             y_token = y[i]
#             if x_token == y_token:
#                 y_.append('<keep>')
#                 if len(x_) == 0:
#                     break
#                 x_token = x_.pop(0)
#             else:
#                 y_.append('<sub_{}>'.format(y_token))
#                 x_token = x_.pop(0)
#                 while True:
#                     y_.append('<delete>')
#                     if x_token == ')':
#                         if len(x_) != 0:
#                             x_token = x_.pop(0)
#                         break
#                     x_token = x_.pop(0)
#         return x, y_
#     # for Arithmetic Operators Restoration (AOR)
#     elif data_src == 'aor':
#         # make a copy
#         y = data
#         x = data.copy()
#         # get operator indexes
#         operator_idxes = [i for i, token in enumerate(y) if not token.isdigit()][::-1]
#         # decide how many operators to remove
#         num_idxes = np.random.choice(range(len(operator_idxes)+1))
#         if num_idxes == 0:
#             return x, ['<keep>']*len(y)
#         else:
#             # decide operators to remove
#             idxes_to_remove = operator_idxes[:num_idxes]
#             x = [x[i] for i in range(len(x)) if i not in idxes_to_remove]
#             # convert to tagging sequences
#             x_ = x.copy()
#             y_ = []
#             x_token = x_.pop(0)
#             for i in range(len(y)):
#                 y_token = y[i]
#                 if x_token == y_token: 
#                     y_.append('<keep>')
#                     if len(x_) == 0:
#                         break
#                     x_token = x_.pop(0)
#                 else:
#                     y_.append('<insert_{}>'.format(y_token))
#             return x, y_
#     # for Arithmetic Equation Correction (AEC)
#     elif data_src == 'aec': 
#         # pick an intermediate step
#         x, y = data
#         xs = [x.copy()] 
#         editops = levenshtein_editops_list(x, y)
#         if len(editops) == 0: 
#             return x, ['<keep>'] * len(x) 
#         else:
#             c = 0 
#             for tag, i, j in editops: 
#                 i += c
#                 if tag == 'replace':
#                     x[i] = y[j]
#                 elif tag == 'delete':
#                     del x[i]
#                     c -= 1
#                 elif tag == 'insert':
#                     x.insert(i, y[j]) 
#                     c += 1
#                 xs.append(x.copy())
#             index = np.random.choice(range(len(xs)))
#             x = xs[index]
#             # convert to a tagging sequence
#             editops = levenshtein_editops_list(x, y)
#             y_ = ['<keep>'] * len(x)
#             c = 0
#             for tag, i, j in editops:
#                 i += c
#                 if tag == 'replace': 
#                     if y_[i] != '<keep>':
#                         y_.insert(i+1, '<sub_{}>'.format(y[j]))
#                         c += 1
#                     else:
#                         y_[i] = '<sub_{}>'.format(y[j])
#                 elif tag == 'delete':
#                     y_[i] = '<delete>'
#                 elif tag == 'insert': 
#                     y_.insert(i, '<insert_{}>'.format(y[j]))
#                     c += 1
#             return x, y_

# def tag_offline_generator(data_src: str, data) -> list:
#     # for Arithmetic Equation Simplification (AES) 
#     if data_src == 'aes': 
#         x, y = data
#         y_ = []
#         x_ = x.copy()
#         x_token = x_.pop(0)
#         for i in range(len(y)):
#             y_token = y[i]
#             if x_token == y_token:
#                 y_.append('<keep>')
#                 if len(x_) == 0:
#                     break
#                 x_token = x_.pop(0)
#             else:
#                 y_.append('<sub_{}>'.format(y_token))
#                 x_token = x_.pop(0)
#                 while True:
#                     y_.append('<delete>')
#                     if x_token == ')':
#                         if len(x_) != 0:
#                             x_token = x_.pop(0)
#                         break
#                     x_token = x_.pop(0)
#         return x, y_
#     # for Arithmetic Operators Insertion (AOI)
#     elif data_src == 'aor': 
#         return data
#     elif data_src == 'aec': 
#         x, y = data 
#         editops = levenshtein_editops_list(x, y)
#         y_ = ['<keep>'] * len(x)
#         c = 0
#         for tag, i, j in editops:
#             i += c
#             if tag == 'replace': 
#                 if y_[i] != '<keep>':
#                     y_.insert(i+1, '<sub_{}>'.format(y[j]))
#                     c += 1
#                 else:
#                     y_[i] = '<sub_{}>'.format(y[j])
#             elif tag == 'delete':
#                 y_[i] = '<delete>'
#             elif tag == 'insert': 
#                 y_.insert(i, '<insert_{}>'.format(y[j]))
#                 c += 1
#         return x, y_

# def data_generator(data, config):
#     # for end2end
#     if config.method == 'e2e':
#         if config.data_mode == 'online': 
#             xs, ys = zip(*[e2e_online_generator(config.data_src, d) for  d in data])
#         elif config.data_mode == 'offline': 
#             xs, ys = zip(*data)
#     # for recurrent inference
#     elif config.method == 'rec': 
#         if config.data_mode == 'online':
#             xs, ys = zip(*[rec_online_generator(config.data_src, d) for d in data])
#         elif config.data_mode == 'offline':
#             xs, ys = zip(*[rec_offline_generator(config.data_src, d) for d in data])
#     # for tagging
#     elif config.method == 'tag':
#         if config.data_mode == 'online':
#             xs, ys = zip(*[tag_online_generator(config.data_src, d) for d in data])
#         elif config.data_mode == 'offline':
#             xs, ys = zip(*[tag_offline_generator(config.data_src, d) for d in data])

#     return xs, ys

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

# def is_int(v):
#     try:
#         int(v)
#         return True
#     except:
#         return False

# def parse_pos(pos):
#     return int(''.join([i for i in pos if i.isdigit()]))

# def one_step_infer(xs, ys_, src_idx2vocab_dict, src_vocab2idx_dict, tgt_idx2vocab_dict, config): 
#     # detach from devices
#     xs = xs.cpu().detach().numpy() 
#     ys_ = torch.argmax(ys_, dim=2).cpu().detach().numpy() 
#     # remove padding idx
#     xs = [rm_idx(x, config.pad_idx) for x in xs] 
#     # print(translate(ys_[0], tgt_idx2vocab_dict))
#     # convert index to vocab
#     xs = [translate(x, src_idx2vocab_dict) for x in xs]
#     ys_ = [translate(y_, tgt_idx2vocab_dict) for y_ in ys_]
#     # mask completed sequences
#     mask = ~(np.array(ys_) == '<done>').all(axis=-1)
#     if not mask.any():
#         done = True
#     else:
#         done = False
#         idxes = np.arange(len(xs))[mask]
#         # inference function for Arithmetic Operators Restoration (AOR) 
#         if config.data_src == 'aor': 
#             # for loop inference 
#             for i in idxes: 
#                 x, y_ = xs[i], ys_[i]
#                 if y_[0].startswith('<pos_') and y_[1] in set(['+', '-', '*', '/', '==']): 
#                     x.insert(parse_pos(y_[0]), y_[1]) 
#         # inference function for Arithmetic Equation Simplification (AES)
#         elif config.data_src == 'aes': 
#             # for loop inference
#             for i in idxes: 
#                 x, y_ = xs[i], ys_[i]
#                 if y_[0].startswith('<pos_') and y_[1].startswith('<pos_') and y_[2].isdigit():
#                     left_idx = parse_pos(y_[0])
#                     right_idx = parse_pos(y_[1])
#                     xs[i] = x[:left_idx] + [y_[2]] + x[right_idx+1:]
#         # inference function for Arithmetic Equation Correction (AEC) 
#         elif config.data_src == 'aec': 
#             # for loop inference
#             for i in idxes: 
#                 x, y_ = xs[i], ys_[i]
#                 if y_[1].startswith('<pos_'):
#                     pos = parse_pos(y_[1])
#                     if y_[0] == '<sub>' and pos in range(len(x)) and y_[2] in src_vocab2idx_dict: 
#                         x[pos] = y_[2]
#                     elif y_[0] == '<delete>' and pos in range(len(x)):
#                         del x[pos]
#                     elif y_[0] == '<insert>' and y_[2] in src_vocab2idx_dict:
#                         x.insert(pos, y_[2])
#                 xs[i] = x
            
#     if config.data_src == 'aor':
#         xs = [torch.Tensor(translate(x, src_vocab2idx_dict)) for x in xs]
#         xs, x_lens = padding(xs, config.L*2)
#     else:
#         xs = [torch.Tensor(translate(x, src_vocab2idx_dict)) for x in xs]
#         xs, x_lens = padding(xs)

#     return xs.to(config.device), x_lens.to(config.device), done

# def rec_infer(xs, x_lens, model, max_infer_step, 
#     src_idx2vocab_dict, src_vocab2idx_dict, tgt_idx2vocab_dict, config, done=False):
#     # recursive inference in valudation and testing
#     # if config.data_src in ['aoi']:
#     if max_infer_step == 0:
#         return xs, x_lens, False
#     else:
#         xs, x_lens, done = rec_infer(xs, x_lens, model, max_infer_step-1, 
#             src_idx2vocab_dict, src_vocab2idx_dict, tgt_idx2vocab_dict, config, done) 
#         if done:
#             return xs, x_lens, done
#         else:
#             ys_ = model(xs, x_lens) 
#             xs, x_lens, done = one_step_infer(xs, ys_, 
#                 src_idx2vocab_dict, src_vocab2idx_dict, tgt_idx2vocab_dict, config)
#             return xs, x_lens, done

# def tag_execute(x, y_):
#     p = []
#     x_ = x.copy()
#     x_token = x_.pop(0)
#     for y_token in y_:
#         if y_token == '<keep>':
#             # keep token
#             p.append(x_token)
#             if len(x_) == 0:
#                 break
#             else:
#                 x_token = x_.pop(0)
#         elif y_token == '<delete>':
#             # delete token
#             if len(x_) == 0:
#                 break
#             else:
#                 x_token = x_.pop(0)
#         elif 'insert' in y_token:
#             # insert token
#             p.append(y_token[8:-1])
#         elif 'sub' in y_token:
#             # substitute token
#             p.append(y_token[5:-1])
#             if len(x_) == 0:
#                 break
#             else:
#                 x_token = x_.pop(0)
#         else:
#             # end symbol or pad symbol
#             break
#     # return prediction
#     return p

# def tag_infer(xs, ys_, src_idx2vocab_dict, src_vocab2idx_dict, tgt_idx2vocab_dict):
#     # convert index to vocab
#     xs = [translate(x, src_idx2vocab_dict) for x in xs]
#     ys_ = [translate(y_, tgt_idx2vocab_dict) for y_ in ys_]
#     preds = [tag_execute(x, y_) for x, y_ in zip(xs, ys_)]
#     # convert vocab to index
#     return [translate(p, src_vocab2idx_dict) for p in preds]