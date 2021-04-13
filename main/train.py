#!/usr/bin/env python
# -*- coding:utf-8 -*-
__author__ = 'Author'
__email__ = 'Email'

import random
from datetime import datetime

from tqdm import tqdm
import torch
from torch.utils import data as torch_data

from config import Config
from src.utils import load, save, pipeline
from src.utils.eva import Evaluater

class Translator(object):
    """docstring for Translator"""
    def __init__(self):
        super(Translator, self).__init__()
        self.config = Config()
        self.initialize()
        self.load_data()
        self.setup_model()

    def initialize(self):
        # for reproducibility
        random.seed(self.config.random_seed)
        torch.manual_seed(self.config.random_seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        # verify devices which can be either cpu or gpu
        self.config.use_gpu = torch.cuda.is_available()
        self.config.device = 'cuda' if self.config.use_gpu else 'cpu'
        # training settings
        self.start_time = datetime.now()
        # self.finished = False 
        self.step, self.epoch = 0, 0
        self.test_log = ['Start Time: {}'.format(self.start_time)]

    def collate_fn(self, data):
        # a customized collate function used in the data loader 
        data.sort(key=len, reverse=True)
        raw_xs, raw_ys = zip(*data)
        xs, ys = pipeline.preprocess(
            raw_xs, raw_ys, self.src_vocab2idx_dict, self.tgt_vocab2idx_dict, self.config)
        xs, x_lens = pipeline.padding(xs)
        ys, _ = pipeline.padding(ys)

        return (raw_xs, raw_ys), (xs, x_lens, ys)

    def load_data(self):
        # load the vocab dictionary and update config
        vocab_dict = load.load_json(self.config.VOCAB_PATH)
        self.src_vocab2idx_dict = vocab_dict['src']
        self.tgt_vocab2idx_dict = vocab_dict['tgt']
        self.src_idx2vocab_dict = {v: k for k, v in self.src_vocab2idx_dict.items()}
        self.tgt_idx2vocab_dict = {v: k for k, v in self.tgt_vocab2idx_dict.items()}
        self.config.PAD_IDX = self.src_vocab2idx_dict[self.config.PAD_TOKEN]
        self.config.BOS_IDX = self.tgt_vocab2idx_dict[self.config.BOS_TOKEN]
        self.config.EOS_IDX = self.tgt_vocab2idx_dict[self.config.EOS_TOKEN]
        self.config.src_vocab_size = len(self.src_vocab2idx_dict)
        self.config.tgt_vocab_size = len(self.tgt_vocab2idx_dict)
        # read data dictionary from json file
        data_dict = load.load_json(self.config.DATA_PATH)
        # train data loader
        train_dataset = pipeline.Dataset(data_dict[self.config.task])
        self.train_generator = torch_data.DataLoader(
              train_dataset
              , batch_size=self.config.batch_size
              , collate_fn=self.collate_fn
              , shuffle=self.config.shuffle
              , num_workers=self.config.num_workers
              , pin_memory=self.config.pin_memory
              , drop_last=self.config.drop_last
              )
        # test data loader
        test_dataset = pipeline.Dataset(data_dict['test'])
        self.test_generator = torch_data.DataLoader(
              test_dataset, 
              batch_size=self.config.batch_size, 
              collate_fn=self.collate_fn, 
              shuffle=False, 
              num_workers=self.config.num_workers, 
              pin_memory=self.config.pin_memory, 
              drop_last=False)
        # update config
        self.config.train_size = len(train_dataset)
        self.config.train_batch = len(self.train_generator)
        self.config.test_size = len(test_dataset)
        self.config.test_batch = len(self.test_generator)

    def load_check_point(self):
        checkpoint_to_load =  torch.load(self.config.SAVE_POINT, map_location=self.config.device) 
        self.step = checkpoint_to_load['step'] 
        self.epoch = checkpoint_to_load['epoch'] 
        model_state_dict = checkpoint_to_load['model'] 
        self.model.load_state_dict(model_state_dict) 
        self.opt.load_state_dict(checkpoint_to_load['optimizer'])

    def setup_model(self): 
        # initialize model weights, optimizer, and loss function
        self.model = pipeline.pick_model(self.config)
        self.model.apply(pipeline.init_parameters)
        self.criterion = torch.nn.NLLLoss(ignore_index=self.config.PAD_IDX)
        self.opt = torch.optim.Adam(self.model.parameters(), lr=self.config.learning_rate)
        if self.config.load_check_point: 
            self.load_check_point()
        self.config.num_parameters = pipeline.count_parameters(self.model)

    def train(self):
        general_info = pipeline.show_config(self.config, self.model)
        self.test_log.append(general_info)
        while self.epoch < self.config.train_epoch:
            print('\nTraining...')
            train_loss, train_iteration = .0, 0
            all_xs, all_ys, all_y_masks, all_ys_ = [], [], [], []
            self.model.train()
            train_generator = tqdm(self.train_generator)
            for raw_data, data in train_generator:
                data = (d.to(self.config.device) for d in data)
                xs, x_lens, ys = data
            #     print(x_lens.cpu().detach().numpy()[0])
            #     print(pipeline.translate(xs.cpu().detach().numpy()[0], self.src_idx2vocab_dict))
            #     print(pipeline.translate(ys.cpu().detach().numpy()[0], self.tgt_idx2vocab_dict))
            #     break
            # break
                ys_ = self.model(xs, x_lens, ys, self.config.teacher_forcing_ratio)
                loss = self.criterion(ys_.reshape(-1, self.config.tgt_vocab_size), ys.reshape(-1))
                # update step
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.clipping_threshold)
                self.opt.step()
                self.opt.zero_grad()
                train_loss += loss.item()
                train_generator.set_description('Loss:{:.4f}'.format(loss.item()))
                # postprocess
                ys_ = torch.argmax(ys_, dim=2).cpu().detach().numpy() # batch_size, max_ys_seq_len
                xs, ys, ys_ = pipeline.post_process(ys_, raw_data, self.tgt_idx2vocab_dict)
                # print(xs[0])
                # print(ys[0])
                # print(ys_[0])
                all_xs += xs
                all_ys += ys 
                all_ys_ += ys_
                train_iteration += 1
                self.step += 1
                # break
            # break
            # evaluation
            loss = train_loss / train_iteration
            eva_matrix = Evaluater(all_ys, all_ys_)
            eva_msg = 'Train Epoch {} Total Step {} Loss:{:.4f} '.format(self.epoch, self.step, loss)
            eva_msg += eva_matrix.eva_msg
            print(eva_msg)
            # random sample to show
            src, tar, pred = random.choice([(x, y, y_) for x, y, y_ in zip(all_xs, all_ys, all_ys_)])
            print(' src: {}\n tgt: {}\n pred: {}'.format(' '.join(src), ' '.join(tar), ' '.join(pred)))
            # break
            # test
            # self.test()
            self.epoch += 1
            # test
            if not self.epoch % self.config.test_epoch:
                self.test()
            # save model
            pipeline.save_model(
                self.step, self.epoch, self.model.state_dict, self.opt.state_dict, self.config.SAVE_POINT)
            # break
        # test
        # self.test()
        # save log
        end_time = datetime.now()
        self.test_log.append('\nEnd Time: {}'.format(end_time))
        self.test_log.append('\nTotal Time: {}'.format(end_time-self.start_time))
        save.save_txt(self.config.LOG_POINT.format('test'), self.test_log)
        # save ref test result
        test_result = ['Src: {}\nTgt: {}\nPred: {}'.format(
            x, y, y_) for x, y, y_ in zip(self.test_src, self.test_tgt, self.test_pred)]
        save.save_txt(self.config.RESULT_POINT.format('test'), test_result)

    def test(self):
        print('\nTesting...')
        # local test
#         checkpoint_to_load =  torch.load(self.config.LOAD_POINT, map_location=self.config.device) 
#         print('Model restored from {}.'.format(self.config.LOAD_POINT))
#         model.load_state_dict(checkpoint_to_load['model'] )
#         # online test
#         # model.load_state_dict(self.model.state_dict())
        all_xs, all_ys, all_ys_ = [], [], []
        test_generator = tqdm(self.test_generator)
        self.model.eval()
        with torch.no_grad():
            for raw_data, data in test_generator:
                data = (d.to(self.config.device) for d in data)
                xs, x_lens, ys = data
                ys_ = self.model(xs, x_lens, ys, self.config.teacher_forcing_ratio)
                # postprocess
                ys_ = torch.argmax(ys_, dim=2).cpu().detach().numpy() # batch_size, max_ys_seq_len
                xs, ys, ys_ = pipeline.post_process(ys_, raw_data, self.tgt_idx2vocab_dict)
                # print(xs[0])
                # print(ys[0])
                # print(ys_[0])
                all_xs += xs
                all_ys += ys 
                all_ys_ += ys_
                # break
        eva_matrix = Evaluater(all_ys, all_ys_)
        eva_msg = 'Test Epoch {} Total Step {} '.format(self.epoch, self.step)
        eva_msg += eva_matrix.eva_msg
        print(eva_msg)
        # record
        self.test_log.append(eva_msg)
        # random sample to show
        src, tar, pred = random.choice([(x, y, y_) for x, y, y_ in zip(all_xs, all_ys, all_ys_)])
        print(' src: {}\n tgt: {}\n pred: {}'.format(' '.join(src), ' '.join(tar), ' '.join(pred)))
        # save test output
        self.test_src = [' '.join(x) for x in all_xs]
        self.test_tgt = [' '.join(y) for y in all_ys]
        self.test_pred = [' '.join(y_) for y_ in all_ys_]

def main():
    # initialize pipeline
    print('Initialize...')
    t = Translator()
    # train
    t.train()

if __name__ == '__main__':
      main()


# # dependency
# # public
# import torch
# torch.manual_seed(0)
# from torch.utils import data as torch_data

# import os
# from tqdm import tqdm
# from datetime import datetime

# # private
# from config import E2EConfig
# from src.utils.eva import Evaluate
# from src.utils.save import *
# from src.utils.load import *
# from src.utils.pipeline import *


# class TextEditor(object):
#     """docstring for TextEditor"""
#     def __init__(self, config):
#         super(TextEditor, self).__init__()
#         self.start_time = datetime.now()
#         self.val_key_metric = float('-inf')
#         self.val_log = ['Start Time: {}'.format(self.start_time)]
#         self.test_log = self.val_log.copy()
#         self.config = config
#         self.step, self.epoch = 0, 0 # training step and epoch
#         self.finished = False # training done flag
#         self.setup_gpu()
#         self.load_vocab()
#         self.load_data()
#         self.setup_model()
#         # data src specific
#         if self.config.data_src == 'aes':
#             self.aes = ArithmeticEquationSimplification(self.config)
#         else:
#             self.aes = None
#         if self.config.data_src == 'aec':
#             self.aec = ArithmeticEquationCorrection(self.config)
#         else: 
#             self.aec = None

#     def setup_gpu(self): 
#         # verify devices which can be either cpu or gpu
#         self.config.use_gpu = torch.cuda.is_available()
#         self.config.device = 'cuda' if self.config.use_gpu else 'cpu'

#     def load_vocab(self):
#         # load the vocab dictionary and update config
#         vocab_dict = load_json(self.config.VOCAB_PATH)
#         self.src_vocab2idx_dict = vocab_dict['src']
#         self.tgt_vocab2idx_dict = vocab_dict['tgt']
#         self.src_idx2vocab_dict = {v: k for k, v in self.src_vocab2idx_dict.items()}
#         self.tgt_idx2vocab_dict = {v: k for k, v in self.tgt_vocab2idx_dict.items()}
#         self.config.pad_idx = self.src_vocab2idx_dict[self.config.pad_symbol]
#         self.config.start_idx = self.tgt_vocab2idx_dict[self.config.start_symbol]
#         self.config.end_idx = self.tgt_vocab2idx_dict[self.config.end_symbol]
#         self.config.src_vocab_size = len(self.src_vocab2idx_dict)
#         self.config.tgt_vocab_size = len(self.tgt_vocab2idx_dict)

#     def train_end2end_collate_fn(self, data): 
#         # a customized collate function used in the data loader 
#         data.sort(key=len, reverse=True)
#         # sampling for many2one task such as aes and aec
#         data = inverse_sampler(data, self.config.data_src, self.aes, self.aec)
#         # sampling intermediate step
#         xs, ys = data_generator(data, self.config) 
#         # convert to index, add end symbol, and save as tensor
#         xs, ys = preprocess(
#             xs, ys, self.src_vocab2idx_dict, self.tgt_vocab2idx_dict, self.config)
#         if self.config.data_mode == 'online' and self.config.data_src == 'aor':
#                 xs, x_lens = padding(xs, self.config.L*2+1)
#         else:
#             xs, x_lens = padding(xs)
#         ys, _ = padding(ys)

#         return xs, x_lens, ys

#     def test_end2end_collate_fn(self, data): 
#         # a customized collate function used in the data loader 
#         data.sort(key=len, reverse=True)
#         xs, ys = zip(*data)
#         # convert to index, add end symbol, and save as tensor
#         xs, ys = preprocess(
#             xs, ys, self.src_vocab2idx_dict, self.tgt_vocab2idx_dict, self.config)
#         if self.config.data_mode == 'online' and self.config.data_src == 'aor':
#                 xs, x_lens = padding(xs, self.config.L*2+1)
#         else:
#             xs, x_lens = padding(xs)
#         ys, _ = padding(ys)

#         return xs, x_lens, ys

#     def load_data(self): 
#         # read data dictionary from json file
#         self.data_dict = load_json(self.config.DATA_PATH)
#         # train data loader
#         if self.config.data_mode == 'online' or self.config.data_src in ['aes', 'aec']: 
#             self.train_dataset = OnlineDataset(data_dict=self.data_dict['train'])
#         else:
#             self.train_dataset = OfflineDataset(data_dict=self.data_dict['train'])
#         self.trainset_generator = torch_data.DataLoader(
#               self.train_dataset, 
#               batch_size=self.config.batch_size, 
#               collate_fn=self.train_end2end_collate_fn, 
#               shuffle=self.config.shuffle, 
#               num_workers=self.config.num_workers, 
#               pin_memory=self.config.pin_memory, 
#               drop_last=self.config.drop_last)
#         # val data loader
#         self.val_dataset = OfflineDataset(data_dict=self.data_dict['val'])
#         self.valset_generator = torch_data.DataLoader(
#               self.val_dataset, 
#               batch_size=self.config.batch_size, 
#               collate_fn=self.test_end2end_collate_fn, 
#               shuffle=False, 
#               num_workers=self.config.num_workers, 
#               pin_memory=self.config.pin_memory, 
#               drop_last=False)
#         # test data loader
#         self.test_dataset = OfflineDataset(data_dict=self.data_dict['test'])
#         self.testset_generator = torch_data.DataLoader(
#               self.test_dataset, 
#               batch_size=self.config.batch_size, 
#               collate_fn=self.test_end2end_collate_fn, 
#               shuffle=False, 
#               num_workers=self.config.num_workers, 
#               pin_memory=self.config.pin_memory, 
#               drop_last=False)
#         # update config
#         self.config.train_size = len(self.train_dataset)
#         self.config.train_batch = len(self.trainset_generator)
#         self.config.val_size = len(self.val_dataset)
#         self.config.val_batch = len(self.valset_generator)
#         self.config.test_size = len(self.test_dataset)
#         self.config.test_batch = len(self.testset_generator)

#     def load_check_point(self):
#         checkpoint_to_load =  torch.load(self.config.LOAD_POINT, map_location=self.config.device) 
#         self.step = checkpoint_to_load['step'] 
#         self.epoch = checkpoint_to_load['epoch'] 
#         model_state_dict = checkpoint_to_load['model'] 
#         self.model.load_state_dict(model_state_dict) 
#         self.opt.load_state_dict(checkpoint_to_load['optimizer'])

#     def setup_model(self): 
#         # initialize model weights, optimizer, and loss function
#         self.model = pick_model(self.config, 'e2e')
#         self.model.apply(init_parameters)
#         self.criterion = torch.nn.NLLLoss(ignore_index=self.config.pad_idx)
#         self.opt = torch.optim.Adam(self.model.parameters(), lr=self.config.learning_rate)
#         if self.config.load_check_point: 
#             self.load_check_point()
#         self.config.num_parameters = count_parameters(self.model)

#     def train(self):
#         general_info = show_config(self.config, self.model)
#         self.val_log.append(general_info)
#         self.test_log.append(general_info)
#         while not self.finished:
#             print('\nTraining...')
#             self.model.train()
#             # training set data loader
#             trainset_generator = tqdm(self.trainset_generator)
#             for data in trainset_generator: 
#                 data = (d.to(self.config.device) for d in data)
#                 xs, x_lens, ys = data
#                 # print(x_lens.cpu().detach().numpy()[0])
#                 # print(translate(xs.cpu().detach().numpy()[0], self.src_idx2vocab_dict))
#                 # print(translate(ys.cpu().detach().numpy()[0], self.tgt_idx2vocab_dict))
#             #     break
#             # break
#                 ys_ = self.model(xs, x_lens, ys, teacher_forcing_ratio=self.config.teacher_forcing_ratio)
#                 loss = self.criterion(ys_.reshape(-1, self.config.tgt_vocab_size), ys.reshape(-1))
#                 # break
#             # break
#                 # update step
#                 loss.backward()
#                 torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.clipping_threshold)
#                 self.opt.step()
#                 self.opt.zero_grad()
#                 self.step += 1
#                 # break 
#             # check progress
#             loss = loss.item()
#             xs = xs.cpu().detach().numpy() # batch_size, max_xs_seq_len
#             ys = ys.cpu().detach().numpy() # batch_size, max_ys_seq_len
#             ys_ = torch.argmax(ys_, dim=2).cpu().detach().numpy() # batch_size, max_ys_seq_len
#             xs, ys, ys_ = post_process(xs, ys, ys_, self.config)
#             # evaluation
#             eva_matrix = Evaluate(self.config, ys, ys_, self.tgt_idx2vocab_dict, True)
#             eva_msg = 'Train Epoch {} Total Step {} Loss:{:.4f} '.format(self.epoch, self.step, loss)
#             eva_msg += eva_matrix.eva_msg
#             print(eva_msg)
#             # random sample to show
#             src, tar, pred = rand_sample(xs, ys, ys_, 
#                 self.src_idx2vocab_dict, self.tgt_idx2vocab_dict, self.tgt_idx2vocab_dict)
#             print(' src: {}\n tgt: {}\n pred: {}'.format(src, tar, pred))
#             # val
#             self.validate()
#             # # test
#             self.test()
#             # early stopping on the basis of validation result
#             if self.val_epoch >= self.config.val_win_size:
#                 # update flag
#                 self.finished = True
#                 # save log
#                 end_time = datetime.now()
#                 self.val_log.append('\nEnd Time: {}'.format(end_time))
#                 self.val_log.append('\nTotal Time: {}'.format(end_time-self.start_time))
#                 save_txt(self.config.LOG_POINT.format('val'), self.val_log)
#                 self.test_log += self.val_log[-2:]
#                 save_txt(self.config.LOG_POINT.format('test'), self.test_log)
#                 # save val result
#                 val_result = ['Src: {}\nTgt: {}\nPred: {}\n\n'.format(
#                     x, y, y_) for x, y, y_ in zip(self.val_src, self.val_tgt, self.val_pred)]
#                 save_txt(self.config.RESULT_POINT.format('val'), val_result)
#                 # save test result
#                 test_result = ['Src: {}\nTgt: {}\nPred: {}\n\n'.format(
#                     x, y, y_) for x, y, y_ in zip(self.test_src, self.test_tgt, self.test_pred)]
#                 save_txt(self.config.RESULT_POINT.format('test'), test_result)

#             self.epoch += 1
#             self.val_epoch += 1

#     def validate(self):
#         print('\nValidating...')
#         all_xs, all_ys, all_ys_ = [], [], []
#         valset_generator = tqdm(self.valset_generator)
#         self.model.eval()
#         with torch.no_grad():
#             for data in valset_generator:
#                 data = (d.to(self.config.device) for d in data)
#                 xs, x_lens, ys = data
#                 ys_ = self.model(xs, x_lens, ys, teacher_forcing_ratio=0.)
#                 xs = xs.cpu().detach().numpy() # batch_size, max_xs_seq_len
#                 ys = ys.cpu().detach().numpy() # batch_size, max_ys_seq_len
#                 ys_ = torch.argmax(ys_, dim=2).cpu().detach().numpy() # batch_size, max_ys_seq_len
#                 xs, ys, ys_ = post_process(xs, ys, ys_, self.config)
#                 all_xs += xs
#                 all_ys += ys 
#                 all_ys_ += ys_
#                 # break
#         # evaluation
#         eva_matrix = Evaluate(self.config, all_ys, all_ys_, self.tgt_idx2vocab_dict)
#         eva_msg = 'Val Epoch {} Total Step {} '.format(self.epoch, self.step)
#         eva_msg += eva_matrix.eva_msg
#         print(eva_msg)
#         # record
#         self.val_log.append(eva_msg)
#         # random sample to show
#         src, tar, pred = rand_sample(all_xs, all_ys, all_ys_, 
#             self.src_idx2vocab_dict, self.tgt_idx2vocab_dict, self.tgt_idx2vocab_dict)
#         print(' src: {}\n tgt: {}\n pred: {}'.format(src, tar, pred))
#         # early stopping
#         if eva_matrix.key_metric > self.val_key_metric:
#             self.val_epoch = 0
#             self.val_key_metric = eva_matrix.key_metric
#             # save model
#             save_check_point(self.step, self.epoch, self.model.state_dict, self.opt.state_dict, self.config.SAVE_POINT)
#         # save test output
#         self.val_src = [' '.join(translate(x, self.src_idx2vocab_dict)) for x in all_xs]
#         self.val_tgt = [' '.join(translate(y, self.tgt_idx2vocab_dict)) for y in all_ys]
#         self.val_pred = [' '.join(translate(y_, self.tgt_idx2vocab_dict)) for y_ in all_ys_]

#     def test(self):
#         print('\nTesting...')
#         model = pick_model(self.config, 'e2e')
#         # local test
#         checkpoint_to_load =  torch.load(self.config.LOAD_POINT, map_location=self.config.device) 
#         print('Model restored from {}.'.format(self.config.LOAD_POINT))
#         model.load_state_dict(checkpoint_to_load['model'] )
#         # online test
#         # model.load_state_dict(self.model.state_dict())
#         all_xs, all_ys, all_ys_ = [], [], []
#         testset_generator = tqdm(self.testset_generator)
#         model.eval()
#         with torch.no_grad():
#             for data in testset_generator:
#                 data = (d.to(self.config.device) for d in data)
#                 xs, x_lens, ys = data
#                 ys_ = model(xs, x_lens, ys, teacher_forcing_ratio=0.)
#                 xs = xs.cpu().detach().numpy() # batch_size, max_xs_seq_len
#                 ys = ys.cpu().detach().numpy() # batch_size, max_ys_seq_len
#                 ys_ = torch.argmax(ys_, dim=2).cpu().detach().numpy() # batch_size, max_ys_seq_len
#                 xs, ys, ys_ = post_process(xs, ys, ys_, self.config)
#                 all_xs += xs
#                 all_ys += ys 
#                 all_ys_ += ys_
#                 # break
#         eva_matrix = Evaluate(self.config, all_ys, all_ys_, self.tgt_idx2vocab_dict)
#         eva_msg = 'Test Epoch {} Total Step {} '.format(self.epoch, self.step)
#         eva_msg += eva_matrix.eva_msg
#         print(eva_msg)
#         # record
#         self.test_log.append(eva_msg)
#         # random sample to show
#         src, tar, pred = rand_sample(all_xs, all_ys, all_ys_, 
#             self.src_idx2vocab_dict, self.tgt_idx2vocab_dict, self.tgt_idx2vocab_dict)
#         print(' src: {}\n tgt: {}\n pred: {}'.format(src, tar, pred))

#         self.test_src = [' '.join(translate(x, self.src_idx2vocab_dict)) for x in all_xs]
#         self.test_tgt = [' '.join(translate(y, self.tgt_idx2vocab_dict)) for y in all_ys]
#         self.test_pred = [' '.join(translate(y_, self.tgt_idx2vocab_dict)) for y_ in all_ys_]

# def main(): 
#     # initial everything
#     te = TextEditor(E2EConfig())
#     # train!
#     te.train()

# if __name__ == '__main__':
#       main()