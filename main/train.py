#!/usr/bin/env python
# -*- coding:utf-8 -*-
__author__ = 'Shining'
__email__ = 'mrshininnnnn@gmail.com'


# built-in
import random
from datetime import datetime
# public
from tqdm import tqdm
import torch
from torch.utils import data as torch_data
# private
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
        self.step, self.epoch = 0, 0
        self.train_log = ['Start Time: {}'.format(self.start_time)]
        self.test_log = self.train_log.copy()

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
        vocab_dict = load.load_json(self.config.VOCAB_JSON)
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
        data_dict = load.load_json(self.config.DATA_JSON)
        # train data loader
        train_dataset = pipeline.Dataset(data_dict['train'])
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
        self.train_log.append(general_info)
        self.test_log.append(general_info)
        while self.epoch < self.config.train_epoch:
            print('\nTraining...')
            train_loss, train_iteration = .0, 0
            all_xs, all_ys, all_ys_ = [], [], []
            self.model.train()
            train_generator = tqdm(self.train_generator)
            for raw_data, data in train_generator:
                data = (d.to(self.config.device) for d in data)
                xs, x_lens, ys = data
                # print(x_lens.cpu().detach().numpy()[0])
                # print(pipeline.translate(xs.cpu().detach().numpy()[0], self.src_idx2vocab_dict))
                # print(pipeline.translate(ys.cpu().detach().numpy()[0], self.tgt_idx2vocab_dict))
                ys_ = self.model(xs, x_lens, ys)
                # break
            # break
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
                ys_ = pipeline.post_process(ys_, self.tgt_idx2vocab_dict, self.config)
                # record
                xs, ys = raw_data
                all_xs += xs
                all_ys += ys 
                all_ys_ += ys_
                train_iteration += 1
                self.step += 1
                # break
            # break
            # evaluation
            loss = train_loss / train_iteration
            eva_matrix = Evaluater(all_ys, all_ys_, self.config)
            eva_msg = 'Train Epoch {} Total Step {} Loss:{:.4f}\n'.format(self.epoch, self.step, loss)
            eva_msg += eva_matrix.eva_msg
            print(eva_msg)
            # record
            self.train_log.append(eva_msg)
            # random sample to show
            src, tar, pred = random.choice([(x, y, y_) for x, y, y_ in zip(all_xs, all_ys, all_ys_)])
            print(' src: {}\n tgt: {}\n pred: {}'.format(' '.join(src), ' '.join(tar), ' '.join(pred)))
            # break
            self.epoch += 1
            # test
            # self.test()
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
        # train
        self.train_log.append('\nEnd Time: {}'.format(end_time))
        self.train_log.append('\nTotal Time: {}'.format(end_time-self.start_time))
        save.save_txt(self.config.LOG_POINT.format('train'), self.train_log)
        # test
        self.test_log.append('\nEnd Time: {}'.format(end_time))
        self.test_log.append('\nTotal Time: {}'.format(end_time-self.start_time))
        save.save_txt(self.config.LOG_POINT.format('test'), self.test_log)
        # save ref test result
        test_result = ['Src: {}\nTgt: {}\nPred: {}'.format(
            x, y, y_) for x, y, y_ in zip(self.test_src, self.test_tgt, self.test_pred)]
        save.save_txt(self.config.RESULT_POINT.format('test'), test_result)

    def test(self):
        print('\nTesting...')
        test_loss, test_iteration = .0, 0
        all_xs, all_ys, all_ys_ = [], [], []
        test_generator = tqdm(self.test_generator)
        self.model.eval()
        with torch.no_grad():
            for raw_data, data in test_generator:
                data = (d.to(self.config.device) for d in data)
                xs, x_lens, ys = data
                # disable teacher forcing
                ys_ = self.model(xs, x_lens, ys)
                loss = self.criterion(ys_.reshape(-1, self.config.tgt_vocab_size), ys.reshape(-1))
                test_loss += loss.item()
                # postprocess
                ys_ = torch.argmax(ys_, dim=2).cpu().detach().numpy() # batch_size, max_ys_seq_len
                ys_ = pipeline.post_process(ys_, self.tgt_idx2vocab_dict, self.config)
                xs, ys = raw_data
                # record
                all_xs += xs
                all_ys += ys 
                all_ys_ += ys_
                test_iteration += 1
                # break
        loss = test_loss / test_iteration
        eva_matrix = Evaluater(all_ys, all_ys_, self.config)
        eva_msg = 'Test Epoch {} Total Step {} Loss:{:.4f}\n'.format(self.epoch, self.step, loss)
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