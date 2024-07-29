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
def val_one_epoch(model_train, train_dataloader, val_dataloader, data_history, optimizer, epoch, args, device):
    val_loss        = 0
    val_accuracy    = 0
    epoch_step_val      = len(val_dataloader)
    scores, labels = [], []
    
    print('Start Validation')
    pbar = tqdm(total=epoch_step_val, desc=f'Epoch {epoch + 1}/{args["all_epoch"]}',postfix=dict,mininterval=0.3)
    model_train.eval()
    with torch.no_grad():   # 使用此方法更为稳妥保证梯度不会记录和更新
        for iteration, batch in enumerate(val_dataloader):
            images, targets = batch
            images  = images.to(device)
            targets = targets.to(device)

            outputs = model_train(images)

            scores.append(outputs)
            labels.append(targets)

            pbar.set_postfix(**{'iteration': iteration})
            pbar.update(1)
    pbar.close()
    
    #-------------------------------------------------#
    #   对预测结果进行初步处理，方便后续高效计算
    #-------------------------------------------------#
    scores = torch.cat(scores,dim=0).cpu().numpy()
    labels = torch.cat(labels,dim=0).cpu().numpy()
    scores = np.array(scores)[:, np.newaxis, :]
    labels = np.array(labels)            
    
    #-------------------------------------------------#
    #   获取训练数据weibull模型
    #-------------------------------------------------#            
    print("Fittting Weibull distribution...")
    _, mavs, dists = compute_train_score_and_mavs_and_dists(args["train_class_num"], train_dataloader, device, model_train)
    categories = list(range(0, args["train_class_num"]))
    weibull_model = fit_weibull(mavs, dists, categories, args["weibull_tail"], "euclidean")

    #-------------------------------------------------#
    #   分别计算三种方法的指标
    #-------------------------------------------------#
    pred_softmax, pred_softmax_threshold, pred_openmax = [], [], []
    score_softmax, score_openmax = [], []            
    for score in scores:
        so, ss = openmax(weibull_model, categories, score,
                         0.5, args["weibull_alpha"], "euclidean")  # openmax_prob, softmax_prob
        pred_softmax_threshold.append(np.argmax(ss) if np.max(ss) >= args["weibull_threshold"] else args["train_class_num"])
        pred_openmax.append(np.argmax(so) if np.max(so) >= args["weibull_threshold"] else args["train_class_num"])
        score_softmax.append(ss)
        score_openmax.append(so)            
    
    #-------------------------------------------------#
    #   计算评价指标
    #-------------------------------------------------#
    print("Evaluation...")
    eval_softmax_threshold = Evaluation(pred_softmax_threshold, labels, score_softmax)
    eval_openmax = Evaluation(pred_openmax, labels, score_openmax)      
    
    softmax_threshold_acc               = eval_softmax_threshold.accuracy
    softmax_threshold_f1_measure        = eval_softmax_threshold.f1_measure
    softmax_threshold_f1_macro          = eval_softmax_threshold.f1_macro
    softmax_threshold_f1_macro_weighted = eval_softmax_threshold.f1_macro_weighted
    softmax_threshold_AUROC             = eval_softmax_threshold.area_under_roc

    openmax_acc                    = eval_openmax.accuracy
    openmax_f1_measure             = eval_openmax.f1_measure
    openmax_f1_macro               = eval_openmax.f1_macro
    openmax_f1_macro_weighted      = eval_openmax.f1_macro_weighted
    openmax_AUROC                  = eval_openmax.area_under_roc
    
    print(f"SoftmaxThreshold accuracy is %.3f" % (softmax_threshold_acc))
    print(f"SoftmaxThreshold F1 is %.3f" % (softmax_threshold_f1_measure))
    print(f"SoftmaxThreshold f1_macro is %.3f" % (softmax_threshold_f1_macro))
    print(f"SoftmaxThreshold f1_macro_weighted is %.3f" % (softmax_threshold_f1_macro_weighted))
    print(f"SoftmaxThreshold area_under_roc is %.3f" % (softmax_threshold_AUROC))
    print(f"_________________________________________")

    print(f"OpenMax accuracy is %.3f" % (openmax_acc))
    print(f"OpenMax F1 is %.3f" % (openmax_f1_measure))
    print(f"OpenMax f1_macro is %.3f" % (openmax_f1_macro))
    print(f"OpenMax f1_macro_weighted is %.3f" % (openmax_f1_macro_weighted))
    print(f"OpenMax area_under_roc is %.3f" % (openmax_AUROC))
    print(f"_________________________________________")
    print('Finish Validation')
    #-------------------------------------------------#
    #   保存训练信息
    #-------------------------------------------------#
    data_history.append_data(epoch,"softmax_t_acc", softmax_threshold_acc)
    data_history.append_data(epoch,"softmax_t_f1_measure", softmax_threshold_f1_measure)
    data_history.append_data(epoch,"softmax_t_f1_macro", softmax_threshold_f1_macro)
    data_history.append_data(epoch,"softmax_t_f1_macro_w", softmax_threshold_f1_macro_weighted)
    data_history.append_data(epoch,"softmax_t_AUROC", softmax_threshold_AUROC)
    data_history.append_data(epoch,"openmax_acc", openmax_acc)
    data_history.append_data(epoch,"openmax_f1_measure", openmax_f1_measure)
    data_history.append_data(epoch,"openmax_f1_macro", openmax_f1_macro)
    data_history.append_data(epoch,"openmax_f1_macro_w", openmax_f1_macro_weighted)
    data_history.append_data(epoch,"openmax_AUROC", openmax_AUROC) 
    
#-------------------------------------------------#
#   main函数
#-------------------------------------------------#
if __name__ == '__main__':
    ...
