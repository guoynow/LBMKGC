# coding:utf-8
from calendar import c
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import os
import time
import sys
import datetime
import ctypes
import json
import numpy as np
import copy
from tqdm import tqdm
from mmkgc.config import Tester, AdvMixTrainer


class LBTrainer(object):

    def __init__(self,
                 model=None,
                 data_loader=None,
                 test_data_loader=None,
                 train_times=1000,
                 alpha=0.5,
                 use_gpu=True,
                 opt_method="sgd"
                ):

        self.work_threads = 8
        self.train_times = train_times

        self.opt_method = opt_method
        self.optimizer = None
        self.lr_decay = 0
        self.weight_decay = 0
        self.alpha = alpha
        self.model = model
        self.data_loader = data_loader
        self.test_data_loader = test_data_loader
        self.use_gpu = use_gpu
        self.batch_size = self.model.batch_size
        self.beta = 0.1

    def train_one_step(self, data):
        self.optimizer.zero_grad()
        loss, p_score, real_embs = self.model({
            'batch_h': self.to_var(data['batch_h'], self.use_gpu),
            'batch_t': self.to_var(data['batch_t'], self.use_gpu),
            'batch_r': self.to_var(data['batch_r'], self.use_gpu),
            'batch_y': self.to_var(data['batch_y'], self.use_gpu),
            'mode': data['mode']
        })
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def calc_gradient_penalty(self, real_data, fake_data):
        batchsize = real_data[0].shape[0]
        alpha = torch.rand(batchsize, 1).cuda()
        inter_h = alpha * real_data[0].detach() + ((1 - alpha) * fake_data[0].detach())
        inter_r = alpha * real_data[1].detach() + ((1 - alpha) * fake_data[1].detach())
        inter_t = alpha * real_data[2].detach() + ((1 - alpha) * fake_data[2].detach())
        inter_h = torch.autograd.Variable(inter_h, requires_grad=True)
        inter_r = torch.autograd.Variable(inter_r, requires_grad=True)
        inter_t = torch.autograd.Variable(inter_t, requires_grad=True)
        inters = [inter_h, inter_r, inter_t]
        scores = self.model.model.cal_score(inters)

        gradients = torch.autograd.grad(
            outputs=scores,
            inputs=inters,
            grad_outputs=torch.ones(scores.size()).cuda(),
            create_graph=True,
            retain_graph=True,
            only_inputs=True
        )[0]
        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * self.beta  # opt.GP_LAMBDA
        return gradient_penalty

    def run(self):
        if self.use_gpu:
            self.model.cuda()

        if self.optimizer is not None:
            pass
        elif self.opt_method == "Adam" or self.opt_method == "adam":
            self.optimizer = optim.Adam(
                self.model.parameters(),
                lr=self.alpha,
                weight_decay=self.weight_decay,
            )
        else:
            raise NotImplementedError
        print("Finish initializing...")

        training_range = tqdm(range(1, self.train_times + 1))
        for epoch in training_range:
            res = 0.0
            for data in self.data_loader: 
                loss = self.train_one_step(data)
                res += loss
            training_range.set_description("Epoch %d | D loss: %f" % (epoch, res))

                
    def set_model(self, model):
        self.model = model

    def to_var(self, x, use_gpu):
        if use_gpu:
            return Variable(torch.from_numpy(x).cuda())
        else:
            return Variable(torch.from_numpy(x))

    def set_use_gpu(self, use_gpu):
        self.use_gpu = use_gpu

    def set_alpha(self, alpha):
        self.alpha = alpha

    def set_lr_decay(self, lr_decay):
        self.lr_decay = lr_decay

    def set_weight_decay(self, weight_decay):
        self.weight_decay = weight_decay

    def set_opt_method(self, opt_method):
        self.opt_method = opt_method

    def set_train_times(self, train_times):
        self.train_times = train_times

    def set_save_steps(self, save_steps, checkpoint_dir=None):
        self.save_steps = save_steps
        if not self.checkpoint_dir:
            self.set_checkpoint_dir(checkpoint_dir)

    def set_checkpoint_dir(self, checkpoint_dir):
        self.checkpoint_dir = checkpoint_dir


