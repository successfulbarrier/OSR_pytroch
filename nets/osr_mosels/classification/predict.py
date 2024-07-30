# !/usr/bin/env python
# -*- coding: utf-8 -*-
# @File    :   predict.py
# @Time    :   2024/07/27 17:30:38
# @Author  :   机灵巢穴_WitNest
# @Version :   1.0
# @Site    :   https://space.bilibili.com/484950664?spm_id_from=333.1007.0.0
# @Desc    :   官方resnet分类,预测脚本

import sys
sys.path.append("/media/lht/LHT/OSR_code/OSR_pytorch")
import torch
from torchvision import transforms
from PIL import Image
import os
import random
import string
import torch.nn.functional as F
import shutil
from datetime import datetime
from nets.backbones import get_model_from_name
from utils.tools import load_weight, set_seed

#-------------------------------------------------#
#   基本参数设定
#-------------------------------------------------#
image_path  = "imgs/2.png"
model_path  = "logs/train/logs_20240729223055/best_epoch_weights.pth"
num_class   = 200
input_shape = [64,64]
# labels = ['cat', 'dog', 'bird', 'horse', 'rabbit', 'turtle', 'fish', 'hamster', 'parrot', 'snake']
labels = range(200)

#-------------------------------------------------#
#   设置随机种子,请一定要设置随机种子
#-------------------------------------------------#
set_seed(42)

#-------------------------------------------------#
#   选择模型
#-------------------------------------------------#
backbone = "mobilenetv2"
if backbone not in ['vit_b_16', 'swin_transformer_tiny', 'swin_transformer_small', 'swin_transformer_base']:
    model = get_model_from_name[backbone](num_classes = num_class, pretrained = False)
else:
    model = get_model_from_name[backbone](input_shape = input_shape, num_classes = num_class, pretrained = False)

#-------------------------------------------------#
#   加载权重
#-------------------------------------------------#
model = load_weight(model, model_path)
print("提示：有未加载的权重是错误的！！！")

# 定义预处理步骤
preprocess = transforms.Compose([
    transforms.Resize(input_shape[0]+2),
    transforms.CenterCrop(input_shape[0]),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def predict_image(image_path):
    # 读取图片
    img = Image.open(image_path).convert('RGB')
    img_tensor = preprocess(img).unsqueeze(0)

    # 模型推理
    with torch.no_grad():
        output = model(img_tensor)

    # 获取预测结果
    predicted = torch.argmax(F.softmax(output, dim=-1), dim=-1)
    predicted_conf  = round(F.softmax(output, dim=-1).squeeze(0)[predicted.item()].item(), 5)
    predicted_class = labels[predicted.item()]

    # 生成随机字符串
    # random_suffix = ''.join(random.choices(string.ascii_letters + string.digits, k=10))
    random_suffix = datetime.now().strftime("%Y%m%d%H%M%S")
    
    # 重命名图片
    new_name = f"{predicted_class}_{predicted_conf}_{random_suffix}.jpg"
    shutil.copy(image_path, 'logs/test/' + new_name)

# 测试
predict_image(image_path)
print("-->推理完成！！！")