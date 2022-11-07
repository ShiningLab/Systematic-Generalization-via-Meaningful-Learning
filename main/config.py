#!/usr/bin/env python
# -*- coding:utf-8 -*-
__author__ = 'Shining'
__email__ = 'ning.shi@ualberta.ca'


# built-in
import os


class Config():
      # config settings
      def __init__(self):
        self.random_seed = 0
        # scan, geography, advising
        self.data = 'scan'
        # experiments for section 3.3
        # exp0_100, exp0_80, exp0_60, exp0_40, exp0_20, exp0_10
        # exp0_10_, exp0_8_, exp0_6_, exp0_4_, exp0_2_, exp0_1_
        # experiments for section 3.4.1 and 3.4.2
        # exp1, exp2, exp31, exp32, exp33
        # experiments for section 4.2
        # exp41 for normal real data
        # exp42 for augmentated real data
        self.exp = 'exp0_1_'
        self.num_primitives = 4
        self.num_synonyms = 10
        if self.data in ['geography', 'advising']:
            self.num_primitives = 4
            self.num_synonyms = 'all'
        if self.exp in ['exp41', 'exp42']:
            self.num_primitives = 'all'
            self.num_synonyms = 'all'
        # bi_lstm_rnn_att, cnn_att, transformer
        self.model_name = 'bi_lstm_rnn_att'
        self.load_check_point = False
        # I/O
        self.CURR_PATH = os.path.dirname(os.path.realpath(__file__))
        self.RESOURCE_PATH = os.path.join(self.CURR_PATH, 'res')
        self.DATA_PATH = os.path.join(self.RESOURCE_PATH, 'data')
        # data dictionary in json file
        self.DATA_JSON = os.path.join(
            self.DATA_PATH, self.data, self.exp, str(self.num_primitives), str(self.num_synonyms), 'data.json')
        # vocab dictionary in json file
        self.VOCAB_JSON = os.path.join(
            self.DATA_PATH, self.data, self.exp, str(self.num_primitives), str(self.num_synonyms), 'vocab.json')
        # path to save and load check point
        self.SAVE_PATH = os.path.join(
            self.RESOURCE_PATH, 'check_points', self.data, self.exp, str(self.num_primitives), str(self.num_synonyms), str(self.random_seed))
        if not os.path.exists(self.SAVE_PATH): os.makedirs(self.SAVE_PATH)
        self.SAVE_POINT = os.path.join(self.SAVE_PATH, '{}.pt'.format(self.model_name))
        if not os.path.exists(self.SAVE_POINT): self.load_check_point = False
        # path to save log
        self.LOG_PATH = os.path.join(
            self.RESOURCE_PATH, 'log', self.data, self.exp, str(self.num_primitives), str(self.num_synonyms), self.model_name, str(self.random_seed))
        if not os.path.exists(self.LOG_PATH): os.makedirs(self.LOG_PATH)
        self.LOG_POINT = os.path.join(self.LOG_PATH,  '{}.txt')
        # path to save output
        self.RESULT_PATH = os.path.join(
            self.RESOURCE_PATH, 'results', self.data, self.exp, str(self.num_primitives), str(self.num_synonyms), self.model_name, str(self.random_seed))
        if not os.path.exists(self.RESULT_PATH): os.makedirs(self.RESULT_PATH)
        self.RESULT_POINT = os.path.join(self.RESULT_PATH, '{}.txt')
        # initialization
        self.BOS_TOKEN = '<s>'
        self.EOS_TOKEN = '</s>'
        self.UNK_TOKEN = '<unk>'
        self.PAD_TOKEN = '<pad>'
        # data loader
        self.batch_size = 128
        if self.data == 'geography':
            self.batch_size = 32
        self.shuffle = True
        self.num_workers = 4
        self.pin_memory = True
        self.drop_last = False
        # train
        self.train_epoch = 320 if self.model_name == 'cnn_att' else 640
        if self.model_name == 'cnn_att' and self.data == 'geography':
            self.learning_rate = 5e-4
        else:
            self.learning_rate = 1e-4
        # test
        self.test_epoch = 32
        # model
        self.en_hidden_size = 512
        self.de_hidden_size = 512
        self.clipping_threshold = 5.
        # embedding
        self.embedding_size = 512
        # dropout
        self.embedding_drop_rate = 0.5
        self.en_drop_rate = 0.5
        self.de_drop_rate = 0.5
        self.pos_encoder_drop_rate = 0.5
        # rnn specific
        self.en_num_layers = 1
        self.de_num_layers = 1
        self.teacher_forcing_ratio = 0.5
        # cnn specific
        self.kernel_size = 3
        self.cnn_en_num_layers = 10
        self.cnn_de_num_layers = 10
        self.cnn_en_max_length = 128
        self.cnn_de_max_length = 312 if self.data == 'advising' and self.exp in ['exp41', 'exp42'] else 256
        # transformer specific
        self.ffnn_dim = 2048
        self.num_heads = 8
        self.tfm_en_num_layers = 2
        self.tfm_de_num_layers = 2