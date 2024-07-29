# !/usr/bin/env python
# -*- coding: utf-8 -*-
# @File    :   tinyimagenet.py
# @Time    :   2024/07/29 16:25:47
# @Author  :   机灵巢穴_WitNest
# @Version :   1.0
# @Site    :   https://space.bilibili.com/484950664?spm_id_from=333.1007.0.0
# @Desc    :   普通分类的TinyImageNet数据集

from torch.utils.data import Dataset, DataLoader
from torchvision import models, utils, datasets, transforms
import numpy as np
import sys
import os
from PIL import Image
import torchvision
import torchvision.transforms as transforms


class TinyImageNet(Dataset):
    def __init__(self, root, train=True, transform=None):
        self.Train = train
        self.root_dir = root
        self.transform = transform
        self.train_dir = os.path.join(self.root_dir, "train")
        self.val_dir = os.path.join(self.root_dir, "val")

        if (self.Train):
            self._create_class_idx_dict_train()
        else:
            self._create_class_idx_dict_val()

        self._make_dataset(self.Train)

        words_file = os.path.join(self.root_dir, "words.txt")
        wnids_file = os.path.join(self.root_dir, "wnids.txt")

        self.set_nids = set()

        with open(wnids_file, 'r') as fo:
            data = fo.readlines()
            for entry in data:
                self.set_nids.add(entry.strip("\n"))

        self.class_to_label = {}
        with open(words_file, 'r') as fo:
            data = fo.readlines()
            for entry in data:
                words = entry.split("\t")
                if words[0] in self.set_nids:
                    self.class_to_label[words[0]] = (words[1].strip("\n").split(","))[0]

    def _create_class_idx_dict_train(self):
        if sys.version_info >= (3, 5):
            classes = [d.name for d in os.scandir(self.train_dir) if d.is_dir()]
        else:
            classes = [d for d in os.listdir(self.train_dir) if os.path.isdir(os.path.join(train_dir, d))]
        
        classes = sorted(classes)
        num_images = 0
        for root, dirs, files in os.walk(self.train_dir):
            for f in files:
                if f.endswith(".JPEG"):
                    num_images = num_images + 1

        self.len_dataset = num_images;

        self.tgt_idx_to_class = {i: classes[i] for i in range(len(classes))}
        self.class_to_tgt_idx = {classes[i]: i for i in range(len(classes))}

    def _create_class_idx_dict_val(self):
        val_image_dir = os.path.join(self.val_dir, "images")
        if sys.version_info >= (3, 5):
            images = [d.name for d in os.scandir(val_image_dir) if d.is_file()]
        else:
            images = [d for d in os.listdir(val_image_dir) if os.path.isfile(os.path.join(train_dir, d))]
        val_annotations_file = os.path.join(self.val_dir, "val_annotations.txt")
        self.val_img_to_class = {}
        set_of_classes = set()
        with open(val_annotations_file, 'r') as fo:
            entry = fo.readlines()
            for data in entry:
                words = data.split("\t")
                self.val_img_to_class[words[0]] = words[1]
                set_of_classes.add(words[1])

        self.len_dataset = len(list(self.val_img_to_class.keys()))
        classes = sorted(list(set_of_classes))
        # self.idx_to_class = {i:self.val_img_to_class[images[i]] for i in range(len(images))}
        self.class_to_tgt_idx = {classes[i]: i for i in range(len(classes))}
        self.tgt_idx_to_class = {i: classes[i] for i in range(len(classes))}

    def _make_dataset(self, Train=True):
        self.images = []
        if Train:
            img_root_dir = self.train_dir
            list_of_dirs = [target for target in self.class_to_tgt_idx.keys()]
        else:
            img_root_dir = self.val_dir
            list_of_dirs = ["images"]

        for tgt in list_of_dirs:
            dirs = os.path.join(img_root_dir, tgt)
            if not os.path.isdir(dirs):
                continue

            for root, _, files in sorted(os.walk(dirs)):
                for fname in sorted(files):
                    if (fname.endswith(".JPEG")):
                        path = os.path.join(root, fname)
                        if Train:
                            item = (path, self.class_to_tgt_idx[tgt])
                        else:
                            item = (path, self.class_to_tgt_idx[self.val_img_to_class[fname]])
                        self.images.append(item)

    def return_label(self, idx):
        return [self.class_to_label[self.tgt_idx_to_class[i.item()]] for i in idx]

    def __len__(self):
        return self.len_dataset

    def __getitem__(self, idx):
        img_path, tgt = self.images[idx]
        with open(img_path, 'rb') as f:
            sample = Image.open(img_path)
            sample = sample.convert('RGB')
        if self.transform is not None:
            sample = self.transform(sample)

        return sample, tgt


#-------------------------------------------------#
#   Tinyimagenet数据集获取类
#-------------------------------------------------#
class Tinyimagenet(object):
    def __init__(self, num_workers):
        #-------------------------------------------------#
        #   数据集路径
        #-------------------------------------------------#
        self.root_path = "/media/lht/LHT/code/datasets/tiny-imagenet"
        #-------------------------------------------------#
        #   数据集类别信息
        #-------------------------------------------------#
        self.class_names = []
        self.num_classes = 200
        
        #-------------------------------------------------#
        #   数据预处理
        #   此处添加了色彩变换减少模型国拟合增强繁华能力
        #   ColorJitter()函数参数说明：
        #   hue：色调变化范围。值为0表示不变化，值为0.1表示随机变化在[-0.1, 0.1]之间。
        #   saturation：饱和度变化范围。值为0表示不变化，值为0.1表示随机变化在[-0.1, 0.1]之间。
        #   brightness：亮度变化范围。值为0表示不变化，值为0.1表示随机变化在[-0.1, 0.1]之间。
        #   contrast：对比度变化范围。值为0表示不变化，值为0.1表示随机变化在[-0.1, 0.1]之间。
        #-------------------------------------------------#
        self.transform_train = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(64, padding=4),
            transforms.ColorJitter(hue=0.1, saturation=0.2, brightness=0.2, contrast=0.1),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        self.transform_val = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
                
        self.num_workers    = num_workers
        #-------------------------------------------------#
        #   获取数据集
        #-------------------------------------------------#
        self.train_dataset  = TinyImageNet(root=self.root_path, train=True, transform=self.transform_train)
        self.val_dataset    = TinyImageNet(root=self.root_path, train=False, transform=self.transform_val)
        
    #-------------------------------------------------#
    #   获取训练和验证的dataloader，因为有时候需要动态调整batch_size
    #-------------------------------------------------#
    def get_dataloader(self, batch_size):
        train_dataloader    = DataLoader(self.train_dataset, batch_size=batch_size, shuffle=True, num_workers=self.num_workers)
        val_dataloader      = DataLoader(self.val_dataset, batch_size=batch_size, shuffle=False, num_workers=self.num_workers)
        return train_dataloader, val_dataloader
    
    #-------------------------------------------------#
    #   保存类别信息
    #-------------------------------------------------#
    def save_class(self, path):
        with open(path, 'w') as f:
            for class_id, class_name in self.train_dataset.tgt_idx_to_class.items():
                f.write(f"{class_id}\t{class_name}\n") 