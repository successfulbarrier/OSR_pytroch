# !/usr/bin/env python
# -*- coding: utf-8 -*-
# @File    :   flower.py
# @Time    :   2024/07/31 11:21:53
# @Author  :   机灵巢穴_WitNest
# @Version :   1.0
# @Site    :   https://space.bilibili.com/484950664?spm_id_from=333.1007.0.0
# @Desc    :   通用的开集分类数据集训练DataLoader

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import os
from PIL import Image
import torchvision.transforms as transforms


class Merge1Dataset(Dataset):
    def __init__(self, root, train=True, train_class_num=9, val_class_num=9, transform=None):
        self.train              = train
        self.root_dir           = root
        self.transform          = transform
        self.train_dir          = os.path.join(self.root_dir, "train")
        self.test_dir           = os.path.join(self.root_dir, "val")
        self.train_class_num    =train_class_num
        self.val_class_num      =val_class_num
        
        if self.train:
            self._create_class_idx_dict_train()
        else:
            self._create_class_idx_dict_test()

        self._make_dataset(self.train)

    def _create_class_idx_dict_train(self):
        classes = [d for d in os.listdir(self.train_dir) if os.path.isdir(os.path.join(self.train_dir, d))]
        classes = sorted(classes)

        #-------------------------------------------------#
        #   获取前n个类别
        #-------------------------------------------------#
        classes = sorted(classes)[:self.train_class_num]
        num_images = 0
        for class_file in classes:
            for f in os.listdir(os.path.join(self.train_dir,class_file)):
                if f.endswith(".JPEG") or f.endswith(".jpg"):
                    num_images = num_images + 1

        self.len_dataset = num_images

        self.tgt_idx_to_class = {i: classes[i] for i in range(len(classes))}
        self.class_to_tgt_idx = {classes[i]: i for i in range(len(classes))}

    def _create_class_idx_dict_test(self):
        classes = [d for d in os.listdir(self.test_dir) if os.path.isdir(os.path.join(self.test_dir, d))]
        classes = sorted(classes)

        num_images = 0
        for root, _, files in os.walk(self.test_dir):
            for f in files:
                if f.endswith(".jpg"):
                    num_images += 1

        self.len_dataset = num_images

        self.tgt_idx_to_class = {i: classes[i] for i in range(len(classes))}
        self.class_to_tgt_idx = {classes[i]: i for i in range(len(classes))}

    def _make_dataset(self, train=True):
        self.images = []
        img_root_dir = self.train_dir if train else self.test_dir
        list_of_dirs = [target for target in self.class_to_tgt_idx.keys()]

        for tgt in list_of_dirs:
            dirs = os.path.join(img_root_dir, tgt)
            if not os.path.isdir(dirs):
                continue

            for root, _, files in sorted(os.walk(dirs)):
                for fname in sorted(files):
                    if fname.endswith(".jpg"):
                        path = os.path.join(root, fname)
                        if train:
                            item = (path, self.class_to_tgt_idx[tgt])
                        else:
                            item = (path, self.class_to_tgt_idx[tgt])
                        self.images.append(item)

    def __len__(self):
        return self.len_dataset

    def __getitem__(self, idx):
        img_path, tgt = self.images[idx]
        with open(img_path, 'rb') as f:
            sample = Image.open(img_path)
            sample = sample.convert('RGB')
        if self.transform is not None:
            sample = self.transform(sample)
        #-------------------------------------------------#
        #   在验证集中将所有大于训练集类别个数的类设置为其他类
        #-------------------------------------------------#
        if self.train == False:
            if tgt >= self.train_class_num:
                tgt = self.train_class_num
        return sample, tgt


#-------------------------------------------------#
#   Tinyimagenet数据集获取类
#-------------------------------------------------#
class Voc2007_merge1_osr(object):
    def __init__(self, num_workers, train_class_num=9, val_class_num=20, input_shape=[128, 128]):
        #-------------------------------------------------#
        #   数据集路径
        #-------------------------------------------------#
        self.root_path = "/media/lht/LHT/code/datasets/voc2007_merge1"
        #-------------------------------------------------#
        #   数据集类别信息
        #-------------------------------------------------#
        self.class_names = ["cat", "dog", "cow", "sheep", "horse", "bicycle", "motorbike", "car", "bus"]
        self.num_classes = train_class_num
        
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
            transforms.Resize(input_shape[0]+10),
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(input_shape[0], padding=4),
            transforms.ColorJitter(hue=0.1, saturation=0.2, brightness=0.2, contrast=0.1),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        self.transform_val = transforms.Compose([
            transforms.Resize(input_shape[0]+2),
            transforms.CenterCrop(input_shape[0]),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
                
        self.num_workers    = num_workers
        #-------------------------------------------------#
        #   获取数据集
        #-------------------------------------------------#
        self.train_dataset  = Merge1Dataset(root=self.root_path, train=True, transform=self.transform_train, train_class_num=train_class_num, val_class_num=val_class_num)
        self.val_dataset    = Merge1Dataset(root=self.root_path, train=False, transform=self.transform_val, train_class_num=train_class_num, val_class_num=val_class_num)
        
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
                
                
#-------------------------------------------------#
#   main函数
#-------------------------------------------------#
if __name__ == '__main__':
    flower = Flower(4)
    print("1111")
