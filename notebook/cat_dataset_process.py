# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.4.2
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# # import 

# +
import os
import os.path as osp
from glob import glob
import shutil

import cv2
import matplotlib.pyplot as plt
import numpy as np

from tqdm import tqdm
from sklearn.model_selection import train_test_split
# -

# # path

base_dir = '../'
input_dir = osp.join(base_dir, 'input')
dataset_dir = osp.join(input_dir, 'cat-dataset')
save_dir = osp.join(input_dir, 'cat')
os.makedirs(save_dir, exist_ok=True)

# # # copy image to one folder

data_names = os.listdir(dataset_dir)
data_names

# 種類によってフォルダが分けられている猫の画像を一つのフォルダにまとめる
image_num = 1
for data_name in tqdm(data_names):
    data_dir = osp.join(dataset_dir, data_name)
    file_names = os.listdir(data_dir)
    file_names = [file_name for file_name in file_names if osp.splitext(file_name)[1] == '.jpg']
    file_names.sort()
    
    for file_name in file_names:
        image_path = osp.join(data_dir, file_name)
        save_image_num = '{:07}.jpg'.format(image_num)
        save_path = osp.join(save_dir, save_image_num)
        
        shutil.copy(image_path, save_path)
        image_num += 1
# # Split and crop data

image_paths = glob(osp.join(save_dir, '*'))
image_paths.sort()
image_paths[0:10]

train_paths, test_paths = train_test_split(image_paths, test_size=100, random_state=0)
print(len(train_paths))
print(len(test_paths))


def random_crop(image, crop_size):
    h, w, _ = image.shape

    top = np.random.randint(0, h - crop_size[0])
    left = np.random.randint(0, w - crop_size[1])

    bottom = top + crop_size[0]
    right = left + crop_size[1]

    image = image[top:bottom, left:right, :]
    return image


def split_save(image_paths, data_type, crop_size=(128, 128), num_aug=1):
    shapes = []
    for image_path in tqdm(image_paths):
        image_name = osp.basename(image_path)
        image = cv2.imread(image_path)
        for aug_n in range(1, num_aug+1):
            image_rsz = random_crop(image, crop_size)
            save_image_name = '{}-{:3}.jpg'.format(image_name, aug_n)
            image_save_path = osp.join(data_save_dir, 'cat_{}'.format(data_type), save_image_name)
            os.makedirs(osp.dirname(image_save_path), exist_ok=True)
            cv2.imwrite(image_save_path, image_rsz)


def split_crop(image_paths, data_type, crop=False, crop_size=(128,128), num_aug=1):
    shapes = []
    for image_path in tqdm(image_paths):
        image_name = osp.basename(image_path)
        file_name = osp.splitext(image_name)[0]
        image = cv2.imread(image_path)
        if (image.shape[0]<=crop_size[0]) | (image.shape[1]<=crop_size[1]):
            print('size problem', image_path)
            continue
        if crop:
            for aug_n in range(1, num_aug+1):
                image_rsz = random_crop(image, crop_size)
                save_image_name = '{}{:03}.jpg'.format(file_name, aug_n)
                image_save_path = osp.join(input_dir, 'cat_{}'.format(data_type), save_image_name)
                os.makedirs(osp.dirname(image_save_path), exist_ok=True)
                cv2.imwrite(image_save_path, image_rsz)
        else:
            save_image_name = '{}.jpg'.format(file_name)
            image_save_path = osp.join(input_dir, 'cat_{}'.format(data_type), save_image_name)
            os.makedirs(osp.dirname(image_save_path), exist_ok=True)
            cv2.imwrite(image_save_path, image)


random_seed = 0
np.random.seed(random_seed)
split_crop(train_paths, 'train', crop=True)
split_crop(test_paths, 'test')





