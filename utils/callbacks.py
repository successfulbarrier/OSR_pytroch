# !/usr/bin/env python
# -*- coding: utf-8 -*-
# @File    :   callbacks.py
# @Time    :   2024/07/27 16:47:19
# @Author  :   机灵巢穴_WitNest
# @Version :   1.0
# @Site    :   https://space.bilibili.com/484950664?spm_id_from=333.1007.0.0
# @Desc    :   记录训练数据

import datetime
import os

import torch
import matplotlib
matplotlib.use('Agg')
import scipy.signal
import numpy as np
from matplotlib import pyplot as plt
from torch.utils.tensorboard import SummaryWriter


class LossHistory():
    def __init__(self, log_dir, model, input_shape):
        self.log_dir    = log_dir
        self.data = {"train_loss":[], "val_loss":[], "lr":[], "train_acc":[], "val_P":[],
                     "val_R":[], "F-measure":[], "AUROC":[]}
        self.writer     = SummaryWriter(self.log_dir)
        try:
            dummy_input     = torch.randn(2, 3, input_shape[0], input_shape[1])
            self.writer.add_graph(model, dummy_input)
        except:
            pass

    def append_data(self, epoch, type, one_data):
        self.data[type].append(one_data)

        with open(os.path.join(self.log_dir, type+".txt"), 'a') as f:
            f.write(str(one_data))
            f.write("\n")

        self.writer.add_scalar(type, one_data, epoch)
        self.draw_plot(type)

    def draw_plot(self, type):
        iters = range(len(self.data[type]))

        plt.figure()
        plt.plot(iters, self.data[type], "red", linewidth = 2, label=type)
        try:
            if len(self.data[type]) < 25:
                num = 5
            else:
                num = 15
            plt.plot(iters, scipy.signal.savgol_filter(self.data[type], num, 3), 'green', linestyle = '--', linewidth = 2, label='smooth train loss')
        except:
            pass

        plt.grid(True)
        plt.xlabel('Epoch')
        plt.ylabel("value")
        plt.title(type)
        plt.savefig(os.path.join(self.log_dir, type+".png"))
        plt.cla()
        plt.close("all")
