#!/usr/bin/env python
# -*- coding:utf-8 -*-

__author__ = 'Shining'
__email__ = 'mrshininnnnn@gmail.com'


import json


# helper functions
def load_txt_to_list(path: str) -> list:
    with open(path, 'r', encoding='utf-8') as f: 
        return f.read().splitlines()

def scan_parser(lines: list) -> list:
    data = [line.split('IN: ')[1].split(' OUT: ') for line in lines]
    xs = [x.split() for (x, _) in data]
    ys = [y.split() for (_, y) in data]
    return xs, ys

def save_json(path: str, data_dict: dict) -> None:
    with open(path, 'w') as f:
        json.dump(data_dict, f, ensure_ascii=False)

# def white_space_tokenizer(str_seq_list: list) -> list:
#     return [str_seq.split(' ') for str_seq in str_seq_list]