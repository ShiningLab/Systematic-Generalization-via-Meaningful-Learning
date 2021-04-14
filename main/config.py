#!/usr/bin/env python
# -*- coding:utf-8 -*-
__author__ = 'Author'
__email__ = 'Email'


import os

# any change on model structure may cause an error
class Config():
      # config settings
      def __init__(self): 
        # data source
        self.exp = 'exp1' # exp1, exp2, exp3
        # self.data_src = 'scan_l12' # scan_l12, scan_l1, scan_l2
        self.num_primitives = 1
        self.num_synonyms = 10
        # self.task = '10'
        self.auto_regressive = True
        # bi_lstm_rnn_att, transformer
        self.model_name = 'bi_lstm_rnn_att'
        self.load_check_point = False
        # I/O
        self.CURR_PATH = os.path.dirname(os.path.realpath(__file__))
        self.RESOURCE_PATH = os.path.join(self.CURR_PATH, 'res')
        self.DATA_PATH = os.path.join(self.RESOURCE_PATH, 'data')
        # data dictionary in json file
        self.DATA_JSON = os.path.join(
            self.DATA_PATH, self.exp, str(self.num_primitives), str(self.num_synonyms), 'data.json')
        # vocab dictionary in json file
        self.VOCAB_JSON = os.path.join(
            self.DATA_PATH, self.exp, str(self.num_primitives), str(self.num_synonyms), 'vocab.json')
        # path to save and load check point
        self.SAVE_PATH = os.path.join(
            self.RESOURCE_PATH, 'check_points', self.exp, str(self.num_primitives), str(self.num_synonyms))
        if not os.path.exists(self.SAVE_PATH): os.makedirs(self.SAVE_PATH)
        self.SAVE_POINT = os.path.join(self.SAVE_PATH, '{}.pt'.format(self.model_name))
        if not os.path.exists(self.SAVE_POINT): self.load_check_point = False
        # path to save log
        self.LOG_PATH = os.path.join(
            self.RESOURCE_PATH, 'log', self.exp, str(self.num_primitives), str(self.num_synonyms), self.model_name)
        if not os.path.exists(self.LOG_PATH): os.makedirs(self.LOG_PATH)
        self.LOG_POINT = os.path.join(self.LOG_PATH,  '{}.txt')
        # path to save output
        self.RESULT_PATH = os.path.join(
            self.RESOURCE_PATH, 'result', self.exp, str(self.num_primitives), str(self.num_synonyms), self.model_name)
        if not os.path.exists(self.RESULT_PATH): os.makedirs(self.RESULT_PATH)
        self.RESULT_POINT = os.path.join(self.RESULT_PATH, '{}.txt')
        # initialization
        self.random_seed = 1
        self.PAD_TOKEN = '<pad>'
        self.BOS_TOKEN = '<s>'
        self.EOS_TOKEN = '</s>'
        # data loader
        self.batch_size = 128
        self.shuffle = True
        self.num_workers = 4
        self.pin_memory = True
        self.drop_last = False
        # train
        self.train_epoch = 640
        # test
        self.test_epoch = 64
        # model
        self.learning_rate = 1e-4
        self.teacher_forcing_ratio = 0.5
        self.clipping_threshold = 5.
        # embedding
        self.embedding_size = 512
        # encoder
        self.en_hidden_size = 512
        self.en_num_layers = 1 
        # decoder
        self.de_hidden_size = 512
        self.de_num_layers = 1
        # dropout
        self.embedding_drop_rate = 0.5
        self.en_drop_rate = 0.5
        self.de_drop_rate = 0.5
        # self.pos_encoder_drop_rate = 0.5
#         # transformer specific dims
#         self.ffnn_dim = 2048
#         self.num_heads = 8
#         self.tfm_en_num_layers = 2
#         self.tfm_de_num_layers = 2