# !/usr/bin/env python
# -*- coding: utf-8 -*-
# @File    :   val.py
# @Time    :   2024/07/27 17:29:43
# @Author  :   机灵巢穴_WitNest
# @Version :   1.0
# @Site    :   https://space.bilibili.com/484950664?spm_id_from=333.1007.0.0
# @Desc    :   官方resnet开集分类,验证脚本
import sys
sys.path.append("/media/lht/LHT/OSR_code/OSR_pytorch")

import os
from tqdm import tqdm
import torch
from torch import nn
import numpy as np
import torch.nn.functional as F

from utils.optimizer import get_lr
from nets.osr.openmax import compute_train_score_and_mavs_and_dists,fit_weibull,openmax
from utils.evaluation import Evaluation


#-------------------------------------------------#
#   验证一轮的脚本
#-------------------------------------------------#
def val_one_epoch(model_train, val_dataloader, data_history, optimizer, epoch, args, device):
    val_loss        = 0
    val_accuracy    = 0
    epoch_step_val      = len(val_dataloader)

    print('Start Validation')
    pbar = tqdm(total=epoch_step_val, desc=f'Epoch {epoch + 1}/{args["all_epoch"]}',postfix=dict,mininterval=0.3)
    model_train.eval()
    with torch.no_grad():   # 使用此方法更为稳妥保证梯度不会记录和更新
        for iteration, batch in enumerate(val_dataloader):
            images, targets = batch
            images  = images.to(device)
            targets = targets.to(device)

            outputs = model_train(images)

            loss_value  = nn.CrossEntropyLoss()(outputs, targets)
            
            val_loss    += loss_value.item()
            accuracy        = torch.mean((torch.argmax(F.softmax(outputs, dim=-1), dim=-1) == targets).type(torch.FloatTensor))
            val_accuracy    += accuracy.item()
            
            pbar.set_postfix(**{'total_loss': val_loss / (iteration + 1),
                                'accuracy'  : val_accuracy / (iteration + 1), 
                                'lr'        : get_lr(optimizer)})
            pbar.update(1)
            
    #-------------------------------------------------#
    #   保存训练信息
    #-------------------------------------------------#
    data_history.append_data(epoch,"val_loss", val_loss / epoch_step_val)
    data_history.append_data(epoch,"val_P", val_accuracy / epoch_step_val)

    
#-------------------------------------------------#
#   main函数
#-------------------------------------------------#
if __name__ == '__main__':
    ...
