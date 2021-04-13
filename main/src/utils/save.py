#!/usr/bin/env python
# -*- coding:utf-8 -*-
__author__ = 'Author'
__email__ = 'Email'

def save_txt(path, line_list):

      with open(path, 'w', encoding='utf-8') as f:
            for line in line_list:
                  f.write(line + '\n')
      f.close()