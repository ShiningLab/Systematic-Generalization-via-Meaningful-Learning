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
        self.data_src = 'scan_l1' # scan_l1, scan_l2
        self.task = '10'
        self.auto_regressive = True
        # bi_lstm_rnn_att, transformer
        self.model_name = 'bi_lstm_rnn_att'
        self.load_check_point = False
        # I/O
        self.CURR_PATH = os.path.dirname(os.path.realpath(__file__))
        self.RESOURCE_PATH = os.path.join(self.CURR_PATH, 'res')
        # data dictionary in json file
        self.DATA_PATH = os.path.join(self.CURR_PATH, 'res/data/', self.exp, self.data_src, 'data.json')
        # vocab dictionary in json file
        self.VOCAB_PATH = os.path.join(self.CURR_PATH, 'res/data/', self.exp, self.data_src,'vocab.json')
        # path to save and load check point
        self.SAVE_PATH = os.path.join(self.RESOURCE_PATH, 'check_points', self.exp, self.data_src, self.task)
        if not os.path.exists(self.SAVE_PATH): os.makedirs(self.SAVE_PATH)
        self.SAVE_POINT = os.path.join(self.SAVE_PATH, '{}.pt'.format(self.model_name))
        if not os.path.exists(self.SAVE_POINT): self.load_check_point = False
        # path to save test log
        self.LOG_PATH = os.path.join(self.RESOURCE_PATH, 'log', self.exp, self.data_src, self.task, self.model_name)
        if not os.path.exists(self.LOG_PATH): os.makedirs(self.LOG_PATH)
        self.LOG_POINT = os.path.join(self.LOG_PATH,  '{}.txt')
        # path to save test output
        self.RESULT_PATH = os.path.join(self.RESOURCE_PATH, 'result', self.exp, self.data_src, self.task, self.model_name)
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

# class E2EConfig(Config):
#     """docstring for E2EConfig"""
#     def __init__(self):
#         super(E2EConfig, self).__init__()
    

# class RecConfig(Config):
#     """docstring for RecConfig"""
#     def __init__(self):
#         super(RecConfig, self).__init__()
#         # define the max inference step
#         if self.data_src == 'aes':
#             self.max_infer_step = self.L
#             self.tgt_seq_len = 3 # start_idx, end_idx, target integer
#         elif self.data_src == 'aor':
#             self.max_infer_step = self.L
#             self.tgt_seq_len = 3 # action, position, target operator
#         elif self.data_src == 'aec': 
#             self.max_infer_step = self.L
#             self.tgt_seq_len = 3 # action, position, target token


# class TagConfig(Config):
#     """docstring for TagConfig"""
#     def __init__(self):
#         super(TagConfig, self).__init__()
#         if self.data_src == 'aes': 
#             # the max decode step depends on the input sequence
#             self.tgt_seq_len = self.L + self.L*6
#         else:
#             self.tgt_seq_len = None
