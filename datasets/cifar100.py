# !/usr/bin/env python
# -*- coding: utf-8 -*-
# @File    :   cifar100.py
# @Time    :   2024/07/26 22:39:24
# @Author  :   机灵巢穴_WitNest
# @Version :   1.0
# @Site    :   https://space.bilibili.com/484950664?spm_id_from=333.1007.0.0
# @Desc    :   cifar100数据集

from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms


#-------------------------------------------------#
#   cifar100数据集获取类
#-------------------------------------------------#
class Cifar100(object):
    def __init__(self, num_workers):
        #-------------------------------------------------#
        #   数据集路径
        #-------------------------------------------------#
        self.root_path = "/media/lht/LHT/code/datasets"
        #-------------------------------------------------#
        #   数据集类别信息
        #-------------------------------------------------#
        self.class_names = [
            'apple', 'aquarium_fish', 'baby', 'bear', 'beaver', 'bed', 'bee', 'beetle', 'bicycle', 'bottle',
            'bowl', 'boy', 'bridge', 'bus', 'butterfly', 'camel', 'can', 'castle', 'caterpillar', 'cattle',
            'chair', 'chimpanzee', 'clock', 'cloud', 'cockroach', 'couch', 'crab', 'crocodile', 'cup', 'dinosaur',
            'dolphin', 'elephant', 'flatfish', 'forest', 'fox', 'girl', 'hamster', 'house', 'kangaroo', 'keyboard',
            'lamp', 'lawn_mower', 'leopard', 'lion', 'lizard', 'lobster', 'man', 'maple_tree', 'motorcycle', 'mountain',
            'mouse', 'mushroom', 'oak_tree', 'orange', 'orchid', 'otter', 'palm_tree', 'pear', 'pickup_truck', 'pine_tree',
            'plain', 'plate', 'poppy', 'porcupine', 'possum', 'rabbit', 'raccoon', 'ray', 'road', 'rocket',
            'rose', 'sea', 'seal', 'shark', 'shrew', 'skunk', 'skyscraper', 'snail', 'snake', 'spider',
            'squirrel', 'streetcar', 'sunflower', 'sweet_pepper', 'table', 'tank', 'telephone', 'television', 'tiger', 'tractor',
            'train', 'trout', 'tulip', 'turtle', 'wardrobe', 'whale', 'willow_tree', 'wolf', 'woman', 'worm'
        ]
        self.num_classes    = 100
        
        #-------------------------------------------------#
        #   数据预处理
        #-------------------------------------------------#
        self.transform_train = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, padding=4),
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
        self.train_dataset  = torchvision.datasets.CIFAR100(root=self.root_path, train=True, download=False, transform=self.transform_train)
        self.val_dataset    = torchvision.datasets.CIFAR100(root=self.root_path, train=False, download=False, transform=self.transform_val)
        
    #-------------------------------------------------#
    #   获取训练和验证的dataloader，因为有时候需要动态调整batch_size
    #-------------------------------------------------#
    def get_dataloader(self, batch_size):
        train_dataloader    = DataLoader(self.train_dataset, batch_size=batch_size, shuffle=True, num_workers=self.num_workers)
        val_dataloader      = DataLoader(self.val_dataset, batch_size=batch_size, shuffle=False, num_workers=self.num_workers)
        return train_dataloader, val_dataloader
    