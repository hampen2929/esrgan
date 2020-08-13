#!/usr/bin/env python
# coding: utf-8
# %% [markdown]
# # import 

# %%
import os
import numpy as np
import os.path as osp
from glob import glob
from tqdm import tqdm

import cv2
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
import pandas as pd
import shutil


# %% [markdown]
# # function

# %%
def read_txt(label_path):
    f = open(label_path)
    data1 = f.read()  # ファイル終端まで全て読んだデータを返す
    f.close()
    lines1 = data1.split(' ') # 改行で区切る(改行文字そのものは戻り値のデータには含まれない)
    return lines1


# %%
def split_save(image_paths, data_type):
    shapes = []
    for image_path in tqdm(image_paths):
        image_name = osp.basename(image_path)
        image_save_path = osp.join(data_split_dir, '{}'.format(data_type), image_name)
        os.makedirs(osp.dirname(image_save_path), exist_ok=True)
        shutil.copy(image_path, image_save_path)


# %% [markdown]
# # constant

# %%
ratio = 0.1

# %%
cols = [
    'image_path', 'r_eye_x', 'r_eye_y', 'l_eye_x', 'l_eye_y', 'mouth_x', 'mouth_y',
    'r_ear1_x', 'r_ear1_y', 'r_ear2_x', 'r_ear2_y', 'r_ear3_x', 'r_ear3_y', 'l_ear1_x',
    'l_ear1_y', 'l_ear2_x', 'l_ear2_y', 'l_ear3_x', 'l_ear3_y',
]


# %%
xcols = [
    'r_eye_x', 'l_eye_x', 'mouth_x', 'r_ear1_x', 'r_ear2_x', 'r_ear3_x',
    'l_ear1_x', 'l_ear2_x', 'l_ear2_y', 'l_ear3_x', 'l_ear3_y',
    ]


# %% [markdown]
# # path

# %%
data_dir = '../input/'
dataset_dir = osp.join(data_dir, 'cat_dataset')
data_save_dir = osp.join(data_dir, 'cat_face_all')
data_split_dir = osp.join(data_dir, 'cat_face')
os.makedirs(data_save_dir, exist_ok=True)


# %%
image_dirs = glob(osp.join(dataset_dir, '*'))
image_dirs.sort()
image_dir = image_dirs[0]

# %% [markdown]
# # label

# %%
labels = []
for image_dir in image_dirs:
    file_names = os.listdir(image_dir)
    file_paths = glob(osp.join(image_dir, '*'))
    file_paths.sort()
    
    image_paths = [f for f in file_paths if osp.splitext(f)[1]=='.jpg']
    label_paths = [f for f in file_paths if osp.splitext(f)[1]=='.cat']
    
    for image_path, label_path in zip(image_paths, label_paths):
        #image = cv2.imread(image_path)
        label = read_txt(label_path)
        label = [int(l) for l in label if l!='']
        label = label[1:19]
                                            
        img_label = [image_path]
        img_label.extend(label)
        labels.append(img_label)


# %%
label_df = pd.DataFrame(labels, columns=cols)

# %% [markdown]
# # bbox

# %%
bboxes = []
for idx, item in label_df.iterrows():
    item_x = item[[c for c in item.index if c.endswith('x')]]
    item_y = item[[c for c in item.index if c.endswith('y')]]
    
    xmin = item_x.min()
    ymin = item_y.min()
    xmax = item_x.max()
    ymax = item_y.max()
    
    bboxes.append([xmin, ymin, xmax, ymax])    


# %%
bbox_df = pd.DataFrame(bboxes, columns=['xmin', 'ymin', 'xmax', 'ymax'])

# %%
label_bbox_df = pd.concat([label_df, bbox_df], axis=1)


# %% [markdown]
# # crop

# %%
for idx, item in label_bbox_df.iterrows():
    image = cv2.imread(item.image_path)
    h, w, _ = image.shape
    xmin = max(0, item.xmin)
    ymin = max(0, item.ymin)
    xmax = min(w, item.xmax)
    ymax = min(h, item.ymax)
    
    width = xmax - xmin
    height = ymax - ymin
    diff = width - height
    if diff > 0:
        ymax = ymax + abs(diff)
    elif diff < 0:
        xmax = xmax + abs(diff)
    else:
        pass
    
    width_2 = xmax - xmin
    height_2 = ymax - ymin
    assert (width_2 == height_2)
    margin = width_2 * ratio
    if (xmin - margin >= 0) and (ymin - margin >= 0) and (xmax + margin <= w) and (ymax + margin <= h):
        xmin = int(xmin - margin)
        ymin = int(ymin - margin)
        xmax = int(xmax + margin)
        ymax = int(ymax + margin)
        
    crop_image = image[ymin:ymax, xmin:xmax]  
    crop_resize_image = cv2.resize(crop_image, dsize=(128, 128))
    image_save_path = osp.join(data_save_dir, osp.basename(item.image_path))
    os.makedirs(osp.dirname(image_save_path), exist_ok=True)
    cv2.imwrite(image_save_path, crop_resize_image)

# %% [markdown]
# # split

# %%
image_paths = glob(osp.join(data_save_dir, '*'))
image_paths.sort()


# %%
train_paths, test_paths = train_test_split(image_paths, test_size=10, random_state=0)
test_paths, demo_paths = train_test_split(test_paths, test_size=1, random_state=0)


# %%
split_save(train_paths, 'train')
split_save(test_paths, 'test')
split_save(demo_paths, 'demo')


# %%
