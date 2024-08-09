# !/usr/bin/env python
# -*- coding: utf-8 -*-
# @File    :   predict.py
# @Time    :   2024/07/27 17:30:38
# @Author  :   机灵巢穴_WitNest
# @Version :   1.0
# @Site    :   https://space.bilibili.com/484950664?spm_id_from=333.1007.0.0
# @Desc    :   开集分类,预测脚本
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
#   基本参数设定
#-------------------------------------------------#
seed = 43
image_path  = "imgs/4.JPEG"
model_path  = "experiment/resnet18-p/best_epoch_weights.pth"
num_class   = 9
input_shape = [128,128]
backbone = "resnet18"
# labels = ['cat', 'dog', 'bird', 'horse', 'rabbit', 'turtle', 'fish', 'hamster', 'parrot', 'snake']
weibull_model = "experiment/resnet18-p/weibull_model.pkl" 
categories = list(range(num_class))
labels = range(num_class+1)
weibull_alpha = 3
weibull_threshold = 0.8

  
#-------------------------------------------------#
#   读取weibull_model
#-------------------------------------------------#
def read_weibull_model(path):
    with open(path, 'rb') as file:
        recovered_variable = pickle.load(file)
    return recovered_variable


#-------------------------------------------------#
#   一轮预测函数
#-------------------------------------------------#
def predict_image(image_path, preprocess, model):
    # 读取图片
    img = Image.open(image_path).convert('RGB')
    img_tensor = preprocess(img).unsqueeze(0)

    # 模型推理
    with torch.no_grad():
        output = model(img_tensor)
    output = np.array(output)
    # 获取预测结果
    weibull_model_ = read_weibull_model(weibull_model)
    so, ss = openmax(weibull_model_, categories, output,
                        0.5, weibull_alpha, "euclidean")  # openmax_prob, softmax_prob
    
    pred_softmax_threshold = np.argmax(ss) if np.max(ss) >= weibull_threshold else num_class
    pred_openmax = np.argmax(so) if np.max(so) >= weibull_threshold else num_class
    
    # 获取预测结果
    pred_softmax_threshold = labels[pred_softmax_threshold]
    pred_openmax = labels[pred_openmax]
    
    # 生成随机字符串
    # random_suffix = ''.join(random.choices(string.ascii_letters + string.digits, k=10))
    random_suffix = datetime.now().strftime("%Y%m%d%H%M%S")
    
    # 重命名图片
    new_name = f"{pred_softmax_threshold}_softmax_{random_suffix}.jpg"
    shutil.copy(image_path, 'logs/test/' + new_name)
    new_name = f"{pred_openmax}_openmax_{random_suffix}.jpg"
    shutil.copy(image_path, 'logs/test/' + new_name)
 
    
#-------------------------------------------------#
#   推理脚本
#-------------------------------------------------#
def openmax_predict():
    #-------------------------------------------------#
    #   设置随机种子,请一定要设置随机种子
    #-------------------------------------------------#
    set_seed(seed)

    #-------------------------------------------------#
    #   选择模型
    #-------------------------------------------------#
    if backbone not in ['vit_b_16', 'swin_transformer_tiny', 'swin_transformer_small', 'swin_transformer_base']:
        model = get_model_from_name[backbone](num_classes = num_class, pretrained = False)
    else:
        model = get_model_from_name[backbone](input_shape = input_shape, num_classes = num_class, pretrained = False)

    #-------------------------------------------------#
    #   加载权重
    #-------------------------------------------------#
    model = load_weight(model, model_path).eval()
    print("提示：有未加载的权重是错误的！！！")

    # 定义预处理步骤
    preprocess = transforms.Compose([
        transforms.Resize(input_shape[0]+2),
        transforms.CenterCrop(input_shape[0]),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    # 测试
    predict_image(image_path, preprocess, model)
    print("-->推理完成！！！")
    
    
#-------------------------------------------------#
#   main函数
#-------------------------------------------------#
if __name__ == '__main__':
    # a = read_weibull_model("logs/train/logs_20240730213241/weibull_model.pkl")
    # b = a
    openmax_predict()