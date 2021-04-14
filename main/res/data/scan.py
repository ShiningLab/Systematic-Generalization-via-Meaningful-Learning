#!/usr/bin/env python
# -*- coding:utf-8 -*-
__author__ = 'Author'
__email__ = 'Email'

import os
import argparse

import numpy as np
from tqdm import trange
from collections import Counter

import utils

class SCAN(object):
    """docstring for SCAN"""
    def __init__(self, args):
        super(SCAN, self).__init__()
        self.args = args
        # verb only without "turn"
        primitive_xs = ['jump', 'run', 'look', 'walk']
        primitive_ys = ['I_JUMP', 'I_RUN', 'I_LOOK', 'I_WALK']
        self.primitive_xs = primitive_xs[:args.num_primitives]
        self.primitive_ys = primitive_ys[:args.num_primitives]
        self.extra_primitive_xs = [p + '_{}'.format(i) for i in range(args.num_synonyms) for p in self.primitive_xs]
        # command in exp3 to involve synonyms
        self.extra_x_format = '{} left'
        self.extra_y_format = 'I_TURN_LEFT {}'
        # self.extra_x_format = '{} twice'
        # self.extra_y_format = '{} {}'
        # load raw scan dataset
        self.get_raw_scan()
        # augment the raw dataset to involve all commands with synonyms
        self.augment_data()
        # prepare the vocabulary dictionary
        self.get_vocab_dicts()
        # put commands with primitives in train set
        # put commands with synonyms in test set
        self.split_train_test()
        # generate the train set
        self.get_exp_data()
        # save output
        self.save()

    def get_raw_scan(self):
        raw_scan = utils.load_txt_to_list(os.path.join('raw', 'scan', 'tasks.txt'))
        scan_raw_xs, scan_raw_ys = utils.scan_parser(raw_scan)
        # remove primitive rules
        # ['look'] -> ['I_LOOK']
        # ['jump'] -> ['I_JUMP']
        # ['run'] -> ['I_RUN']
        # ['walk'] -> ['I_WALK']
        idxes_list = [i for i, x in enumerate(scan_raw_xs) if len(x) > 1]
        # for i in range(len(scan_raw_xs)):
        #   if i not in idxes_list:
        #       print(scan_raw_xs[i], scan_raw_ys[i])
        # 20906
        self.scan_raw_xs = np.array(scan_raw_xs, dtype=object)[idxes_list].tolist()
        # 20906
        self.scan_raw_ys = np.array(scan_raw_ys, dtype=object)[idxes_list].tolist()

    def augment_data(self):
        self.scan_xs = self.scan_raw_xs.copy()
        self.scan_ys = self.scan_raw_ys.copy()
        for n in trange(self.args.num_synonyms):
            for p in self.primitive_xs:
                new_p = p + '_{}'.format(n)
                for x, y in zip(self.scan_raw_xs, self.scan_raw_ys):
                    if p in x:
                        new_x = ' '.join(x).replace(p, new_p).split()
                        self.scan_xs.append(new_x)
                        self.scan_ys.append(y)

    def get_vocab_dicts(self):
        # source vocabulary
        scan_src_counter = Counter()
        for x in self.scan_xs:
            scan_src_counter.update(x)
        # soruce vocabulary dictionary
        src_vocab2idx_dict = {}
        # to pad sequence length
        src_vocab2idx_dict['<pad>'] = 0
        i = len(src_vocab2idx_dict)
        for token in scan_src_counter:
            src_vocab2idx_dict[token] = i
            i += 1
        print('Source vocab size', len(src_vocab2idx_dict))
        # target vocabulary
        scan_tgt_counter = Counter()
        for y in self.scan_ys:
            scan_tgt_counter.update(y)
        # target vocabulary dictionary
        tgt_vocab2idx_dict = {}
        # to pad sequence length
        tgt_vocab2idx_dict['<pad>'] = 0
        # to mark the start of a sequence
        tgt_vocab2idx_dict['<s>'] = 1
        # to mark the end of a sequence
        tgt_vocab2idx_dict['</s>'] = 2

        i = len(tgt_vocab2idx_dict)
        for token in scan_tgt_counter:
            tgt_vocab2idx_dict[token] = i
            i += 1
        print('Target vocab size', len(tgt_vocab2idx_dict))
        self.vocab_dict = {}
        self.vocab_dict['src'] = src_vocab2idx_dict
        self.vocab_dict['tgt'] = tgt_vocab2idx_dict

    def split_train_test(self):
        train_idxes = []
        test_idxes = []
        for i, x in enumerate(self.scan_xs):
            if set(self.extra_primitive_xs) & set(x):
                test_idxes.append(i)
            else:
                train_idxes.append(i)
        self.train_xs = np.array(self.scan_xs, dtype=object)[train_idxes].tolist()
        self.train_ys = np.array(self.scan_ys, dtype=object)[train_idxes].tolist()
        print('Base train size', len(self.train_xs))
        self.test_xs = np.array(self.scan_xs, dtype=object)[test_idxes].tolist()
        self.test_ys = np.array(self.scan_ys, dtype=object)[test_idxes].tolist()
        print('Base test size', len(self.test_xs))

    def get_exp_data(self):
        test_dict = {}
        test_dict['xs'] = self.test_xs
        test_dict['ys'] = self.test_ys
        self.data_dict = {}
        self.data_dict['test'] = test_dict
        train_dict = {}
        train_xs = self.train_xs.copy()
        train_ys = self.train_ys.copy()
        for i in range(self.args.num_synonyms):
            for p_x, p_y in zip(self.primitive_xs, self.primitive_ys):
                new_p_x = p_x + '_{}'.format(i)
                if self.args.exp in [1, 2]:
                    train_xs += [[new_p_x]] * self.args.num_oversample
                    train_ys += [[p_y]] * self.args.num_oversample    
                elif self.args.exp == 3:
                    train_xs += [self.extra_x_format.format(new_p_x).split()] * self.args.num_oversample
                    train_ys += [self.extra_y_format.format(p_y).split()] * self.args.num_oversample
                else:
                    print('Check --exp')
        if self.args.exp == 1:
            train_xs += [[x] for x in self.primitive_xs]
            train_ys += [[y] for y in self.primitive_ys]
        train_dict['xs'] = train_xs
        train_dict['ys'] = train_ys
        self.data_dict['train'] = train_dict
        print('Exp {} Train size'.format(self.args.exp), len(self.data_dict['train']['xs']))
        print('Exp {} Test size'.format(self.args.exp), len(self.data_dict['test']['xs']))

    def save(self):
        # save output as json
        out_path = os.path.join('exp' + str(self.args.exp), str(self.args.num_primitives), str(self.args.num_synonyms))
        if not os.path.exists(out_path): os.makedirs(out_path)
        utils.save_json(os.path.join(out_path, 'data.json'), self.data_dict)
        utils.save_json(os.path.join(out_path, 'vocab.json'), self.vocab_dict)

def main():
    # example
    # python scan.py --exp 1 --num_primitives 1 --num_synonyms 10 --num_oversample 1
    # parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp', 
        type=int, 
        required=True, 
        help='defines for which experiment to generate data')
    parser.add_argument('--num_primitives', 
        type=int, 
        required=True, 
        help='defines the number of primitives from jump, run, look and walk')
    parser.add_argument('--num_synonyms', 
        type=int, 
        required=True, 
        help='defines the number of synonyms for each primitive to generalize')
    parser.add_argument('--num_oversample', 
        type=int, 
        required=True, 
        help='defines the number of primitive rules to oversample during training')
    args = parser.parse_args()
    scan = SCAN(args)

if __name__ == '__main__': 
    main()