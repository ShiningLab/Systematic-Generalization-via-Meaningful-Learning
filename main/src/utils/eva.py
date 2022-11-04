#!/usr/bin/env python
# -*- coding:utf-8 -*-
__author__ = 'Shining'
__email__ = 'ning.shi@ualberta.ca'


# public
import numpy as np


class Evaluater(object):
    """docstring for Evaluater"""
    def __init__(self, ys, ys_, config):
        super(Evaluater, self).__init__()
        self.ys, self.ys_ = ys, ys_
        self.size = len(ys)
        self.token_acc = self.get_token_acc()
        self.seq_acc = self.get_seq_acc()
        # generate an evaluation message
        self.eva_msg = 'Token Acc:{:.4f} Seq Acc:{:.4f}'.format(self.token_acc, self.seq_acc)

    def check_token(self, y, y_):
        min_len = min([len(y), len(y_)])
        max_len = max([len(y), len(y_)])
        return np.float32(sum(np.equal(np.array(y[:min_len], dtype=object), np.array(y_[:min_len], dtype=object))) / max_len)

    def check_seq(self, y, y_): 
        min_len = min([len(y), len(y_)])
        max_len = max([len(y), len(y_)])
        if sum(np.equal(np.array(y[:min_len], dtype=object), np.array(y_[:min_len]), dtype=object)) == max_len: 
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