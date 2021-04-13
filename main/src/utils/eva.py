#!/usr/bin/env python
# -*- coding:utf-8 -*-
__author__ = 'Author'
__email__ = 'Email'


import numpy as np

class Evaluater(object):
    """docstring for Evaluater"""
    def __init__(self, ys, ys_):
        super(Evaluater, self).__init__()
        self.ys, self.ys_ = ys, ys_
        self.size = len(ys)
        self.token_acc = self.get_token_acc()
        self.seq_acc = self.get_seq_acc()
        # generate an evaluation message
        self.eva_msg = 'Token Acc:{:.4f} Seq Acc:{:.4f}'.format(self.token_acc, self.seq_acc)

    def check_token(self, y, y_):
        min_len = min([len(y), len(y_)])
        return np.float32(sum(np.equal(np.array(y[:min_len], dtype=object), np.array(y_[:min_len], dtype=object))) / len(y))

    def check_seq(self, y, y_): 
        min_len = min([len(y), len(y_)]) 
        if sum(np.equal(np.array(y[:min_len], dtype=object), np.array(y_[:min_len]), dtype=object)) == len(y): 
            return 1
        return 0

    def get_token_acc(self):
        a = 0
        for i in range(self.size):
            y = self.ys[i]
            y_ = self.ys_[i]
            a += self.check_token(y, y_)
        return np.float32(a/self.size)

    def get_seq_acc(self): 
        a = 0
        for i in range(self.size):
            y = self.ys[i]
            y_ = self.ys_[i]
            a += self.check_seq(y, y_)
        return np.float32(a/self.size)

#         self.config = config
#         self.idx2vocab_dict = idx2vocab_dict
#         self.tars = targets
#         self.preds = predictions
#         self.size = len(targets)
#         # token-level accuracy
#         self.token_acc = self.get_token_acc()
#         # sequence-level accuracy
#         self.seq_acc = self.get_seq_acc()
#         # main metric for early stopping
#         self.key_metric = self.token_acc
#         # generate an evaluation message
#         self.eva_msg = 'Token Acc:{:.4f} Seq Acc:{:.4f}'.format(self.token_acc, self.seq_acc)
#         if self.config.data_src in ['aor', 'aec'] and not train:
#             # if hold equation 
#             self.eq_acc = self.get_eq_acc()
#             # main metric for early stopping
#             self.key_metric = self.eq_acc
#             # generate an evaluation message
#             self.eva_msg += ' Equation Acc:{:.4f}'.format(self.eq_acc)

#     def check_equation(self, tgt, pred): 
#         # remove end symbol
#         if self.config.method == 'e2e':
#             # remove end symbol
#             tgt = [t for t in tgt if t != self.config.end_idx]
#             pred = [p for p in pred if p != self.config.end_idx]
#         # e.g., ['3', '-', '3', '+', '9', '-', '3', '==', '6']
#         tgt = [self.idx2vocab_dict[t] for t in tgt]
#         # e.g., ['3', '-', '3', '+', '9', '+', '3', '==', '6']
#         pred = [self.idx2vocab_dict[p] for p in pred]
#         # e.g., ['3', '3', '9', '3', '6']
#         tgt_nums = [t for t in tgt if t.isdigit()]
#         # e.g., ['3', '3', '9', '3', '6']
#         pred_nums = [p for p in pred if p.isdigit()]
#         # eval('123') return 123
#         # eval('1 2 3') raise error
#         try:
#             if tgt_nums == pred_nums and pred[-1].isdigit() and pred[-2] == '==':
#                     right = int(pred[-1])
#                     left = eval(' '.join(pred[:-2]))
#                     if left == right:
#                         return 1
#         except:
#             return 0
#         return 0





#     def get_eq_acc(self):
#         a = 0
#         for i in range(self.size):
#             tar = self.tars[i]
#             pred = self.preds[i]
#             a += self.check_equation(tar, pred)
#         return np.float32(a/self.size)



