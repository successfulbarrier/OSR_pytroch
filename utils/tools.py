# !/usr/bin/env python
# -*- coding: utf-8 -*-
# @File    :   tools.py
# @Time    :   2024/07/26 21:59:08
# @Author  :   机灵巢穴_WitNest
# @Version :   1.0
# @Site    :   https://space.bilibili.com/484950664?spm_id_from=333.1007.0.0
# @Desc    :   用到的一些小工具，难以具体分类

import yaml
import shutil
import os
from datetime import datetime
import torch
import random
import numpy as np

#-------------------------------------------------#
#   设定随机种子
#-------------------------------------------------#
def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)
#-------------------------------------------------#
#   读取配置文件
#-------------------------------------------------#
def read_yaml_config(file_path):
    current_time = datetime.now().strftime("%Y%m%d%H%M%S")
    #-------------------------------------------------#
    #   创建文件夹
    #-------------------------------------------------#
    folder_name = f"logs/train/logs_{current_time}"
    os.makedirs(folder_name)
    #-------------------------------------------------#
    #   备份配置文件
    #-------------------------------------------------#
    new_file_path = f"{folder_name}/cfg_{current_time}.yaml"
    shutil.copyfile(file_path, new_file_path)
    #-------------------------------------------------#
    #   读取配置文件
    #-------------------------------------------------#
    with open(file_path, 'r') as file:
        config = yaml.safe_load(file)
    
    config["train_output_path"] = folder_name
    return config

#-------------------------------------------------#
#   权重初始化
#-------------------------------------------------#
def weights_init(net, init_type='normal', init_gain=0.02):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and classname.find('Conv') != -1:
            if init_type == 'normal':
                torch.nn.init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                torch.nn.init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                torch.nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                torch.nn.init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
        elif classname.find('BatchNorm2d') != -1:
            torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
            torch.nn.init.constant_(m.bias.data, 0.0)
    print('initialize network with %s type' % init_type)
    net.apply(init_func)
    
    
#-------------------------------------------------#
#   加载模型权重
#-------------------------------------------------#
def load_weight(model, model_path):
        print('Load weights {}.'.format(model_path))
        model_dict      = model.state_dict()
        pretrained_dict = torch.load(model_path, map_location = torch.device('cpu'))
        load_key, no_load_key, temp_dict = [], [], {}
        for k, v in pretrained_dict.items():
            if k in model_dict.keys() and np.shape(model_dict[k]) == np.shape(v):
                temp_dict[k] = v
                load_key.append(k)
            else:
                no_load_key.append(k)
        model_dict.update(temp_dict)
        model.load_state_dict(model_dict)
        print("\nSuccessful Load Key:", str(load_key)[:500], "……\nSuccessful Load Key Num:", len(load_key))
        print("\nFail To Load Key:", str(no_load_key)[:500], "……\nFail To Load Key num:", len(no_load_key))
        print("\n\033[1;33;44m温馨提示，head部分没有载入是正常现象，Backbone部分没有载入是错误的。\033[0m")
        return model
    
    
#-------------------------------------------------#
#   测试
#-------------------------------------------------#
if __name__ == '__main__':
    # 读取yaml配置文件
    config = read_yaml_config('cfgs/default.yaml')
