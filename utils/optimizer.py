# !/usr/bin/env python
# -*- coding: utf-8 -*-
# @File    :   operter.py
# @Time    :   2024/07/27 16:15:29
# @Author  :   机灵巢穴_WitNest
# @Version :   1.0
# @Site    :   https://space.bilibili.com/484950664?spm_id_from=333.1007.0.0
# @Desc    :   优化器和学习率相关的函数

import torch.optim as optim
import math
from functools import partial

#-------------------------------------------------#
#   获取优化器
#-------------------------------------------------#
def get_optimizer(model_train, args):
    nbs             = 64
    lr_limit_max    = 1e-3 if args["optimizer"] == 'adam' else 1e-1
    lr_limit_min    = 1e-4 if args["optimizer"] == 'adam' else 5e-4
    if args["backbone"] in ['vit_b_16', 'swin_transformer_tiny', 'swin_transformer_small', 'swin_transformer_base']:
        nbs             = 256
        lr_limit_max    = 1e-3 if args["optimizer"] == 'adam' else 1e-1
        lr_limit_min    = 1e-5 if args["optimizer"] == 'adam' else 5e-4
    if args["freeze_epoch"] != 0:
        Init_lr_fit     = min(max(args["freeze_batch_size"] / nbs * args["init_lr"], lr_limit_min), lr_limit_max)
        Min_lr_fit      = min(max(args["freeze_batch_size"] / nbs * args["min_lr"], lr_limit_min * 1e-2), lr_limit_max * 1e-2)
    else:
        Init_lr_fit     = min(max(args["unfreeze_batch_size"] / nbs * args["init_lr"], lr_limit_min), lr_limit_max)
        Min_lr_fit      = min(max(args["unfreeze_batch_size"] / nbs * args["min_lr"], lr_limit_min * 1e-2), lr_limit_max * 1e-2)        
    
    optimizer = {
        'adam'  : optim.Adam(model_train.parameters(), Init_lr_fit, betas = (args["momentum"], 0.999), weight_decay=args["weight_decay"]),
        'sgd'   : optim.SGD(model_train.parameters(), Init_lr_fit, momentum = args["momentum"], nesterov=True)
    }[args["optimizer"]]
    return optimizer, Init_lr_fit, Min_lr_fit

#-------------------------------------------------#
#   获取学习率调整函数
#-------------------------------------------------#
def get_lr_scheduler(lr_decay_type, lr, min_lr, total_iters, warmup_iters_ratio = 0.05, warmup_lr_ratio = 0.1, no_aug_iter_ratio = 0.05, step_num = 10):
    def yolox_warm_cos_lr(lr, min_lr, total_iters, warmup_total_iters, warmup_lr_start, no_aug_iter, iters):
        if iters <= warmup_total_iters:
            # lr = (lr - warmup_lr_start) * iters / float(warmup_total_iters) + warmup_lr_start
            lr = (lr - warmup_lr_start) * pow(iters / float(warmup_total_iters), 2) + warmup_lr_start
        elif iters >= total_iters - no_aug_iter:
            lr = min_lr
        else:
            lr = min_lr + 0.5 * (lr - min_lr) * (
                1.0 + math.cos(math.pi* (iters - warmup_total_iters) / (total_iters - warmup_total_iters - no_aug_iter))
            )
        return lr
    
    def step_lr(lr, decay_rate, step_size, iters):
        if step_size < 1:
            raise ValueError("step_size must above 1.")
        n       = iters // step_size
        out_lr  = lr * decay_rate ** n
        return out_lr

    if lr_decay_type == "cos":
        warmup_total_iters  = min(max(warmup_iters_ratio * total_iters, 1), 3)
        warmup_lr_start     = max(warmup_lr_ratio * lr, 1e-6)
        no_aug_iter         = min(max(no_aug_iter_ratio * total_iters, 1), 15)
        func = partial(yolox_warm_cos_lr ,lr, min_lr, total_iters, warmup_total_iters, warmup_lr_start, no_aug_iter)
    else:
        decay_rate  = (min_lr / lr) ** (1 / (step_num - 1))
        step_size   = total_iters / step_num
        func = partial(step_lr, lr, decay_rate, step_size)

    return func


#-------------------------------------------------#
#   调整学习率
#-------------------------------------------------#
def set_optimizer_lr(optimizer, lr_scheduler_func, epoch):
    lr = lr_scheduler_func(epoch)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
        
        
#---------------------------------------------------#
#   获得学习率
#---------------------------------------------------#
def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']