# !/usr/bin/env python
# -*- coding: utf-8 -*-
# @File    :   train.py
# @Time    :   2024/07/26 21:23:05
# @Author  :   机灵巢穴_WitNest
# @Version :   1.0
# @Site    :   https://space.bilibili.com/484950664?spm_id_from=333.1007.0.0
# @Desc    :   官方resnet开集分类,训练脚本
import sys
sys.path.append("/media/lht/LHT/OSR_code/OSR_pytorch")

import os
from tqdm import tqdm
import torch
from torch import nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

from utils.tools import read_yaml_config, weights_init, load_weight, set_seed
from nets.backbones import get_model_from_name
from datasets import get_dataset_from_name
from utils.callbacks import LossHistory
from utils.optimizer import get_optimizer, get_lr_scheduler, set_optimizer_lr, get_lr
from nets.osr_mosels.openmax_osr.val import val_one_epoch
from nets.osr.openmax import weibull_model_output


#-------------------------------------------------#
#   一轮训练脚本
#-------------------------------------------------#
def train_one_epoch(model_train, train_dataloader, data_history, optimizer, epoch, args, device):
    total_loss      = 0
    total_accuracy  = 0

    epoch_step      = len(train_dataloader)
    
    print('Start Train')
    pbar = tqdm(total=epoch_step,desc=f'Epoch {epoch + 1}/{args["all_epoch"]}',postfix=dict,mininterval=0.3)
    model_train.train()
    for iteration, batch in enumerate(train_dataloader):
        if iteration >= epoch_step: 
            break
        images, targets = batch

        images  = images.to(device)
        targets = targets.to(device)
                
        #----------------------#
        #   清零梯度
        #----------------------#
        optimizer.zero_grad()

        #----------------------#
        #   前向传播
        #----------------------#
        outputs     = model_train(images)
        #----------------------#
        #   计算损失
        #----------------------#
        loss_value  = nn.CrossEntropyLoss()(outputs, targets)
        loss_value.backward()
        optimizer.step()

        total_loss += loss_value.item()
        with torch.no_grad():
            accuracy = torch.mean((torch.argmax(F.softmax(outputs, dim=-1), dim=-1) == targets).type(torch.FloatTensor))
            total_accuracy += accuracy.item()
            pbar.set_postfix(**{'total_loss': total_loss / (iteration + 1), 
                                'accuracy'  : total_accuracy / (iteration + 1), 
                                'lr'        : get_lr(optimizer)})
            pbar.update(1)

    pbar.close()
    print('Finish Train')
    #-------------------------------------------------#
    #   保存训练信息
    #-------------------------------------------------#
    data_history.append_data(epoch,"train_loss", total_loss/epoch_step)
    data_history.append_data(epoch,"train_acc", total_accuracy/epoch_step)
    data_history.append_data(epoch,"lr", get_lr(optimizer))
    
#-------------------------------------------------#
#   训练脚本
#-------------------------------------------------#
def train(args):
    #-------------------------------------------------#
    #   设置随机种子,请一定要设置随机种子
    #-------------------------------------------------#
    set_seed(args["seed"])
    #-------------------------------------------------#
    #   设置训练设备，分类一本比较小用不上多卡训练，故只采用单卡训练
    #-------------------------------------------------#
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    #-------------------------------------------------#
    #   选择数据集
    #-------------------------------------------------#
    dataset = get_dataset_from_name[args["dataset"]](num_workers=args["num_workers"], train_class_num=args["train_class_num"], 
                                                     val_class_num=args["val_class_num"])
    dataset.save_class(os.path.join(args["train_output_path"],"classes.txt"))
    if args["freeze_epoch"] != 0:
        train_dataloader, val_dataloader = dataset.get_dataloader(batch_size=args["freeze_batch_size"])
    else:
        train_dataloader, val_dataloader = dataset.get_dataloader(batch_size=args["unfreeze_batch_size"])
    
    #-------------------------------------------------#
    #   选择模型
    #-------------------------------------------------#
    backbone = args["backbone"]
    if backbone not in ['vit_b_16', 'swin_transformer_tiny', 'swin_transformer_small', 'swin_transformer_base']:
        model = get_model_from_name[backbone](num_classes = dataset.num_classes, pretrained = args["pretrained"])
    else:
        model = get_model_from_name[backbone](input_shape = args["input_shape"], num_classes = dataset.num_classes, pretrained = args["pretrained"])
    # print(model)
    #-------------------------------------------------#
    #   没有使用预训练权重，并且也没有加载权重则对权重进行初始化
    #   目前仅对卷积层和BN层进行了初始化
    #-------------------------------------------------#
    if not args["pretrained"]:
        weights_init(model)
    #-------------------------------------------------#
    #   加载权重
    #-------------------------------------------------#
    if args["model_path"] != None:
        model = load_weight(model, args["model_path"])
    else:
        print("-->本次训练未加载预训练权重！！！")
   
    #-------------------------------------------------#
    #   如果是导出weibull_model，到此处结束
    #-------------------------------------------------#
    if args["weibull_model_output"]:
        model = model.to(device).eval()
        weibull_model_output(model, train_dataloader, device, args)
        return 0
    #-------------------------------------------------#
    #   训练数据记录
    #-------------------------------------------------#
    loss_history = LossHistory(args["train_output_path"], model, input_shape=args["input_shape"])    
    loss_history.data = {"train_loss":[], "lr":[], "train_acc":[], 
                         "softmax_t_acc":[],"softmax_t_f1_measure":[], "softmax_t_f1_macro":[], 
                         "softmax_t_f1_macro_w":[], "softmax_t_AUROC":[],
                         "openmax_acc":[],"openmax_f1_measure":[], "openmax_f1_macro":[], 
                         "openmax_f1_macro_w":[], "openmax_AUROC":[]}
    
    #-------------------------------------------------#
    #   将模型设置进入训练模式
    #-------------------------------------------------#
    model_train = torch.nn.DataParallel(model).to(device).train()
    cudnn.benchmark = True
    
    #-------------------------------------------------#
    #   冻结模型训练
    #-------------------------------------------------#
    if args["freeze_epoch"] != 0:
        model.freeze_backbone()
        UnFreeze_flag = False
        print("-->开始冻结训练！！！")
    else:
        UnFreeze_flag = True
        print("-->本次训练未冻结训练！！！")
        
    #-------------------------------------------------#
    #   获取优化器
    #-------------------------------------------------#
    optimizer, Init_lr_fit, Min_lr_fit = get_optimizer(model_train, args)
    
    #-------------------------------------------------#
    #   获取学习率下降函数
    #-------------------------------------------------#
    lr_scheduler_func = get_lr_scheduler(args["lr_decay_type"], Init_lr_fit, Min_lr_fit, args["all_epoch"])
    
    #---------------------------------------#
    #   开始模型训练
    #---------------------------------------#
    for epoch in range(args["all_epoch"]):
        #-------------------------------------------------#
        #   如果模型在进行冻结训练，大于冻结轮数之后解冻
        #-------------------------------------------------#    
        if epoch >= args["freeze_epoch"] and args["freeze_epoch"] != 0 and UnFreeze_flag == False:
            #-------------------------------------------------------------------#
            #   判断当前batch_size，自适应调整学习率
            #-------------------------------------------------------------------#
            nbs             = 64
            lr_limit_max    = 1e-3 if args["optimizer"] == 'adam' else 1e-1
            lr_limit_min    = 1e-4 if args["optimizer"] == 'adam' else 5e-4
            if backbone in ['vit_b_16', 'swin_transformer_tiny', 'swin_transformer_small', 'swin_transformer_base']:
                nbs             = 256
                lr_limit_max    = 1e-3 if args["optimizer"] == 'adam' else 1e-1
                lr_limit_min    = 1e-5 if args["optimizer"] == 'adam' else 5e-4
            Init_lr_fit     = min(max(args["unfreeze_batch_size"] / nbs * args["init_lr"], lr_limit_min), lr_limit_max)
            Min_lr_fit      = min(max(args["unfreeze_batch_size"] / nbs * args["min_lr"], lr_limit_min * 1e-2), lr_limit_max * 1e-2)
            #---------------------------------------#
            #   获得学习率下降的公式
            #---------------------------------------#
            lr_scheduler_func = get_lr_scheduler(args["train_output_path"], Init_lr_fit, Min_lr_fit, args["all_epoch"])
            #-------------------------------------------------#
            #   调整batch_size
            #-------------------------------------------------#
            train_dataloader, val_dataloader = dataset.get_dataloader(batch_size=args["unfreeze_batch_size"])
            #-------------------------------------------------#
            #   解冻模型
            #-------------------------------------------------#
            model.Unfreeze_backbone()
            UnFreeze_flag = True
            print("-->解除冻结训练！！！")
            
        
        #-------------------------------------------------#
        #   调整学习率
        #-------------------------------------------------#
        set_optimizer_lr(optimizer, lr_scheduler_func, epoch)
        
        #-------------------------------------------------#
        #   训练一轮
        #-------------------------------------------------#
        train_one_epoch(model_train, train_dataloader, loss_history, optimizer, epoch, args, device)
        
        #-------------------------------------------------#
        #   验证一轮,这里需要传入训练集，建立weibull模型
        #-------------------------------------------------#
        if (epoch+1) % 5 == 0 and epoch!=0:
            val_one_epoch(model_train, train_dataloader, val_dataloader, loss_history, optimizer, epoch, args, device)
        
        #-------------------------------------------------#
        #   保存最优权重
        #-------------------------------------------------#
        if len(loss_history.data["softmax_t_acc"]) <= 1 or loss_history.data["softmax_t_acc"][-1] >= max(loss_history.data["softmax_t_acc"][:-1]):
            print('Save best model to best_epoch_weights.pth')
            torch.save(model.state_dict(), os.path.join(args["train_output_path"], "best_epoch_weights.pth"))
            
        torch.save(model.state_dict(), os.path.join(args["train_output_path"], "last_epoch_weights.pth"))
        
#-------------------------------------------------#
#   main函数
#-------------------------------------------------#
if __name__ == '__main__':
    #-------------------------------------------------#
    #   获取参数
    #-------------------------------------------------#
    args = read_yaml_config("cfgs/default_osr.yaml")
    
    #-------------------------------------------------#
    #   开始训练
    #-------------------------------------------------#
    train(args)