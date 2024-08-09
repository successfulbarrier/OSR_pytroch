# !/usr/bin/env python
# -*- coding: utf-8 -*-
# @File    :   det_classify.py
# @Time    :   2024/08/02 10:29:18
# @Author  :   机灵巢穴_WitNest
# @Version :   1.0
# @Site    :   https://space.bilibili.com/484950664?spm_id_from=333.1007.0.0
# @Desc    :   实现对检测网络预测的结果进行细分类

import sys
sys.path.append("/media/lht/LHT/OSR_code/OSR_pytorch")

import os
from tqdm import tqdm
import torch
import pickle
from torchvision import transforms
from PIL import Image
import numpy as np
import string
import torch.nn.functional as F
import shutil
from datetime import datetime
from nets.backbones import get_model_from_name
from utils.tools import load_weight, set_seed
from nets.osr.openmax import openmax


#-------------------------------------------------#
#   基础参数
#-------------------------------------------------#
seed                = 42
det_result_path     = "/media/lht/LHT/DL_code/yolov8-pytorch-my/experiment/yolov8-n-merge1-1.5/detection-results"
output_result_path  = "/media/lht/LHT/DL_code/yolov8-pytorch-my/map_out/detection-results"
image_path          = "/media/lht/LHT/code/datasets/VOCdevkit/VOC2007/JPEGImages"
model_path          = "experiment/swim-t-p/best_epoch_weights.pth"
input_shape         = [128,128]
backbone            = "swin_transformer_tiny"

merger_class_name   = ["quadrupeds", "two_car", "four_car"]
class_name          = ["bicycle", "bus", "car", "cat", "cow", "dog", "horse", "motorbike", "sheep"]
class_know_num      = len(class_name)
class_idx           = list(range(class_know_num))

weibull_model       = "experiment/swim-t-p/weibull_model.pkl" 
weibull_alpha       = 3
weibull_threshold   = 0.7

#-------------------------------------------------#
#   读取weibull_model
#-------------------------------------------------#
def read_weibull_model(path):
    with open(path, 'rb') as file:
        recovered_variable = pickle.load(file)
    return recovered_variable


#-------------------------------------------------#
#   细分类函数
#-------------------------------------------------#
def det_classify_func():
    #-------------------------------------------------#
    #   设置随机种子,请一定要设置随机种子
    #-------------------------------------------------#
    set_seed(seed)

    #-------------------------------------------------#
    #   设置训练设备，分类一本比较小用不上多卡训练，故只采用单卡训练
    #-------------------------------------------------#
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    #-------------------------------------------------#
    #   选择模型
    #-------------------------------------------------#
    if backbone not in ['vit_b_16', 'swin_transformer_tiny', 'swin_transformer_small', 'swin_transformer_base']:
        model = get_model_from_name[backbone](num_classes = class_know_num, pretrained = False)
    else:
        model = get_model_from_name[backbone](input_shape = input_shape, num_classes = class_know_num, pretrained = False)

    #-------------------------------------------------#
    #   加载权重
    #-------------------------------------------------#
    model = load_weight(model, model_path).eval()
    print("提示：有未加载的权重是错误的！！！")

    #-------------------------------------------------#
    #   定义图像预处理步骤
    #-------------------------------------------------#
    preprocess = transforms.Compose([
        transforms.Resize(input_shape[0]+2),
        transforms.CenterCrop(input_shape[0]),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    
    #-------------------------------------------------#
    #   加载weibull模型
    #-------------------------------------------------#
    weibull_model_ = read_weibull_model(weibull_model)
    
    #-------------------------------------------------#
    #   获取文件列表
    #-------------------------------------------------#
    file_list = os.listdir(det_result_path)
    os.makedirs(output_result_path, exist_ok=True)
    
    #-------------------------------------------------#
    #   循环调整每个文件
    #-------------------------------------------------#
    for det_file in tqdm(file_list):
        one_det_classify(model, det_file, preprocess, weibull_model_, device)
        
    
    print("-->调整完毕！！！")


#-------------------------------------------------#
#   细分类单张图片
#-------------------------------------------------#
def one_det_classify(model, det_file, preprocess, weibull_model_, device):
    #-------------------------------------------------#
    #   读取检测结果
    #-------------------------------------------------#
    with open(os.path.join(det_result_path, det_file), 'r', encoding='utf-8') as f:
        lines = f.readlines()

    #-------------------------------------------------#
    #   挑出需要进行细分类的类别
    #-------------------------------------------------#
    merged_lines = []
    un_merged_lines = []
    for line in lines:
        parts = line.strip().split(' ')
        if parts[0] in merger_class_name:
            merged_lines.append(parts)
        else:
            un_merged_lines.append(parts)
    
    #-------------------------------------------------#
    #   如果未有需要细分的类，则直接返回
    #-------------------------------------------------#
    if merged_lines == []:
        return 0
    
    #-------------------------------------------------#
    #   读取图片
    #-------------------------------------------------#
    img = Image.open(os.path.join(image_path, det_file[:-4]+".jpg")).convert('RGB')
    
    #-------------------------------------------------#
    #   获取需要进行细分类类别的图片数据
    #-------------------------------------------------#
    img_part_list = []
    for parts in merged_lines:
        img_part = img.crop((int(parts[2]), int(parts[3]), int(parts[4]), int(parts[5])))
        img_part = preprocess(img_part)
        img_part_list.append(img_part)
    input_image = torch.stack(img_part_list, dim=0).to(device)
    
    #-------------------------------------------------#
    #   进行网络推理
    #-------------------------------------------------#
    model.eval()
    model = model.to(device)
    with torch.no_grad():
        output = model(input_image)
    
    #-------------------------------------------------#
    #   后处理
    #-------------------------------------------------#
    new_merged_lines = []
    outputs = np.array(output.to("cpu"))[:, np.newaxis, :]   #转化为numpy类型的数据
    for i, output in enumerate(outputs):
        so, ss = openmax(weibull_model_, class_idx, output,
                            0.5, weibull_alpha, "euclidean")  # openmax_prob, softmax_prob
        
        pred_softmax_threshold = np.argmax(ss) if np.max(ss) >= weibull_threshold else class_know_num
        pred_openmax = np.argmax(so) if np.max(so) >= weibull_threshold else class_know_num
        if pred_openmax != class_know_num:
            merged_lines[i][0] = class_name[pred_openmax]
            new_merged_lines.append(merged_lines[i])
            
    #-------------------------------------------------#
    #   保存结果
    #-------------------------------------------------#
    full_lines = new_merged_lines + un_merged_lines             
    with open(os.path.join(output_result_path, det_file), 'w') as f:
        for line in full_lines:
            f.write(' '.join(line) + '\n')
            
#-------------------------------------------------#
#   main函数
#-------------------------------------------------#
if __name__ == '__main__':
    det_classify_func()
    
