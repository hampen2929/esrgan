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

import sys
import os
import os.path as osp
import math
import json

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable
from torchvision.models import vgg19
from torchvision.utils import save_image
import torchvision.transforms as transforms

from glob import glob

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython import display
from IPython.display import clear_output

from PIL import Image

import time
import datetime


# # args

class opts():
    pass


opt = opts()

# +
opt.channles = 3
opt.hr_height = 128
opt.residual_blocks = 23
opt.lr = 0.0002
opt.b1 = 0.9
opt.b2 = 0.999
opt.batch_size = 4
opt.n_cpu = 8
opt.n_epoch = 200
opt.warmup_batches = 500
# opt.warmup_batches = 5
opt.lambda_adv = 5e-3
opt.lambda_pixel = 1e-2

opt.pretrained = False

opt.dataset_name = 'cat'
# opt.dataset_name = 'img_align_celeba_resize'
# opt.dataset_name = 'img_align_celeba_resize'

opt.sample_interval = 50
opt.checkpoint_interval = 100
# -

args = [arg for arg in dir(opt) if not arg.startswith('__')]

opt_dict = {arg: getattr(opt, arg) for arg in args}

hr_shape = (opt.hr_height, opt.hr_height)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

mean = np.array([0.485, 0.456, 0.406])
std = np.array([0.229, 0.224, 0.225])


# # model

# # Generator

class DenseResidualBlock(nn.Module):
    def __init__(self, filters, res_scale=0.2):
        super(DenseResidualBlock, self).__init__()
        self.res_scale = res_scale
        
        def block(in_features, non_linearity=True):
            layers = [nn.Conv2d(in_features, filters, 3, 1, 1, bias=True)]
            if non_linearity:
                layers += [nn.LeakyReLU()]
            return nn.Sequential(*layers)
    
        self.b1 = block(in_features=1 * filters)
        self.b2 = block(in_features=2 * filters)
        self.b3 = block(in_features=3 * filters)
        self.b4 = block(in_features=4 * filters)
        self.b5 = block(in_features=5 * filters, non_linearity=False)
        self.blocks = [self.b1, self.b2, self.b3, self.b4, self.b5]
    
    def forward(self, x):
        inputs = x
        for block in self.blocks:
            out = block(inputs)
            inputs = torch.cat([inputs, out], 1)
        return out.mul(self.res_scale) + x


class ResidualInResidualDenseBlock(nn.Module):
    def __init__(self, filters, res_scale=0.2):
        super(ResidualInResidualDenseBlock, self).__init__()
        self.res_scale = res_scale
        self.dense_blocks = nn.Sequential(
            DenseResidualBlock(filters), DenseResidualBlock(filters), DenseResidualBlock(filters)
        )
    
    def forward(self, x):
        return self.dense_blocks(x).mul(self.res_scale) + x


class GeneratorPRDB(nn.Module):
    def __init__(self, channels, filters=64, num_res_blocks=16, num_upsample=2):
        super(GeneratorPRDB, self).__init__()
        
        self.conv1 = nn.Conv2d(channels, filters, kernel_size=3, stride=1, padding=1)
        
        self.res_blocks = nn.Sequential(*[ResidualInResidualDenseBlock(filters) for _ in range(num_res_blocks)])
        self.conv2 = nn.Conv2d(filters, filters, kernel_size=3, stride=1, padding=1)
        
        upsample_layers = []
        
        for _ in range(num_upsample):
            upsample_layers += [
                nn.Conv2d(filters, filters * 4, kernel_size=3, stride=1, padding=1),
                nn.LeakyReLU(),
                nn.PixelShuffle(upscale_factor=2),
            ]
        self.upsampling = nn.Sequential(*upsample_layers)
        self.conv3 = nn.Sequential(
            nn.Conv2d(filters, filters, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(filters, channels, kernel_size=3, stride=1, padding=1),
        )
    
    def forward(self, x):
        out1 = self.conv1(x)
        out = self.res_blocks(out1)
        out2 = self.conv2(out)
        out = torch.add(out1, out2)
        out = self.upsampling(out)
        out = self.conv3(out)
        return out


# # Discriminator

class FeatureExtractor(nn.Module):
    def __init__(self):
        super(FeatureExtractor, self).__init__()
        vgg19_model = vgg19(pretrained=True)
        self.vgg19_54 = nn.Sequential(*list(vgg19_model.features.children())[:35])

    def forward(self, img):
        return self.vgg19_54(img)


class Discriminator(nn.Module):
    def __init__(self, input_shape):
        super(Discriminator, self).__init__()
                
        self.input_shape = input_shape
        in_channels, in_height, in_width = self.input_shape
        patch_h, patch_w = int(in_height / 2 ** 4), int(in_width / 2 ** 4)
        self.output_shape = (1, patch_h, patch_w)
    
        def descriminator_block(in_filters, out_filters, first_block=False):
            layers = []
            layers.append(nn.Conv2d(in_filters, out_filters, kernel_size=3, stride=1, padding=1))
            if not first_block:
                layers.append(nn.BatchNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            layers.append(nn.Conv2d(out_filters, out_filters, kernel_size=3, stride=2, padding=1))
            layers.append(nn.BatchNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers
        
        layers = []
        in_filters = in_channels
        for i, out_filters in enumerate([64, 128, 256, 512]):
            print(descriminator_block(in_filters, out_filters, first_block=(i == 0)))
            layers.extend(descriminator_block(in_filters, out_filters, first_block=(i == 0)))
            in_filters = out_filters
        
        layers.append(nn.Conv2d(out_filters, 1, kernel_size=3, stride=1, padding=1))
        
        self.model = nn.Sequential(*layers)
    
    def forward(self, img):
        return self.model(img)


# ## Dataset

def denormalize(tensors):
    for c in range(3):
        tensors[:, c].mul_(std[c]).add_(mean[c])
    return torch.clamp(tensors, 0, 255)


class ImageDataset(Dataset):
    def __init__(self, dataset_dir, hr_shape):
        hr_height, hr_width = hr_shape
        
        self.lr_transform = transforms.Compose([
            transforms.Resize((hr_height // 4, hr_height // 4), Image.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)])

        self.hr_transform = transforms.Compose([
            transforms.Resize((hr_height, hr_height), Image.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)])
        
        self.files = sorted(glob(osp.join(dataset_dir, '*')))
    
    def __getitem__(self, index):
        img = Image.open(self.files[index % len(self.files)])
        img_lr = self.lr_transform(img)
        img_hr = self.hr_transform(img)
        
        return {'lr': img_lr, 'hr': img_hr}
    
    def __len__(self):
        return len(self.files)


class TestImageDataset(Dataset):
    def __init__(self, dataset_dir):
        # TODO: 入力に対して1/4
        self.hr_transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean, std)])
        self.files = sorted(glob(osp.join(dataset_dir, '*')))
    
    def lr_transform(self, img, img_size):
        img_width, img_height = img_size
        self.__lr_transform = transforms.Compose([
            transforms.Resize((img_height // 4, img_width // 4), Image.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)])
        img = self.__lr_transform(img)
        return img
            
    def __getitem__(self, index):
        img = Image.open(self.files[index % len(self.files)])
        img_size = img.size
        img_lr = self.lr_transform(img, img_size)
        img_hr = self.hr_transform(img)
        
        return {'lr': img_lr, 'hr': img_hr}
    
    def __len__(self):
        return len(self.files)


def save_json(label, save_path):
    f = open(save_path, "w")
    json.dump(label, f, ensure_ascii=False, indent=4, 
              sort_keys=True, separators=(',', ': '))


# # path

ROOT = '../'

input_dir = osp.join(ROOT, 'input')
output_dir = osp.join(ROOT, 'output', str(datetime.datetime.fromtimestamp(time.time())))
weight_dir = osp.join(ROOT, 'weight')

# +
image_train_save_dir = osp.join(output_dir, 'image', 'train')
image_test_save_dir = osp.join(output_dir, 'image', 'test')
weight_save_dir = osp.join(output_dir, 'weight')

os.makedirs(image_train_save_dir, exist_ok=True)
os.makedirs(image_test_save_dir, exist_ok=True)
os.makedirs(weight_save_dir, exist_ok=True)
# -

train_data_dir = osp.join(input_dir, '{}_train'.format(opt.dataset_name))
test_data_dir = osp.join(input_dir, '{}_test_sub2'.format(opt.dataset_name))
g_weight_path = osp.join(weight_dir, 'generator.pth')
d_weight_path = osp.join(weight_dir, 'discriminator.pth')

opt_save_path = osp.join(output_dir, 'opt.json')

save_json(opt_dict, opt_save_path)

# ## set_model

# +
generator = GeneratorPRDB(opt.channles, filters=64, num_res_blocks=opt.residual_blocks).to(device)
discriminator = Discriminator(input_shape=(opt.channles, *hr_shape)).to(device)

if opt.pretrained:
    generator.load_state_dict(torch.load(g_weight_path))
    discriminator.load_state_dict(torch.load(d_weight_path))

feature_extractor = FeatureExtractor().to(device)
feature_extractor.eval()
# -

# # Loss

criterion_GAN = nn.BCEWithLogitsLoss().to(device)
criterion_content = nn.L1Loss().to(device)
criterion_pixel = nn.L1Loss().to(device)

# # Optimizer

optimizer_G = optim.Adam(generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
optimizer_D = optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

# # Tensor

Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.Tensor

# # dataset

# +
train_dataloader = DataLoader(
    ImageDataset(train_data_dir, hr_shape=hr_shape),
    batch_size=opt.batch_size,
    shuffle=True,
    num_workers=opt.n_cpu,
)

test_dataloader = DataLoader(
    TestImageDataset(test_data_dir),
    batch_size=1,
    shuffle=False,
    num_workers=opt.n_cpu,
)

# +
# test_dataloader = DataLoader(
#     TestImageDataset('../input/sample/'),
#     batch_size=1,
#     shuffle=False,
#     num_workers=opt.n_cpu,
# )
# -

# # main

# +
loss_names = ['batch_num', 'loss_pixel', 'loss_D', 'loss_G', 'loss_content', 'loss_GAN']
train_infos = []

plt.figure(figsize=(16,9))
low_image_save = False

for epoch in range(1, opt.n_epoch + 1):
    for batch_num, imgs in enumerate(train_dataloader):
        batches_done = (epoch - 1) * len(train_dataloader) + batch_num
        
        # preprocess
        imgs_lr = Variable(imgs['lr'].type(Tensor))
        imgs_hr = Variable(imgs['hr'].type(Tensor))
        
        # ground truth
        valid = Variable(Tensor(np.ones((imgs_lr.size(0), *discriminator.output_shape))), 
                         requires_grad=False)
        fake = Variable(Tensor(np.zeros((imgs_lr.size(0), *discriminator.output_shape))), 
                        requires_grad=False)
        
        # バックプロパゲーションの前に勾配を０にする
        optimizer_G.zero_grad()
        
        # 低解像度の画像から高解像度の画像を生成
        gen_hr = generator(imgs_lr)
        
        loss_pixel = criterion_pixel(gen_hr, imgs_hr)
        
        # 画素単位の損失であるloss_pixelで事前学習を行う
        if batches_done <= opt.warmup_batches:
            loss_pixel.backward()
            optimizer_G.step()
            train_info = {
                'epoch': epoch, 
                'batch_num': batch_num,
                'loss_pixel': loss_pixel.item()
            }
        
            sys.stdout.write('\r{}'.format('\t'*10))
            sys.stdout.write('\r {}'.format(train_info))            
        else:
        
            # prediction
            pred_real = discriminator(imgs_hr).detach()
            pred_fake = discriminator(gen_hr)

            # Aeversarial loss
            loss_GAN = criterion_GAN(pred_fake - pred_real.mean(0, keepdim=True), valid)

            # content loss(perceptual loss)
            # 特徴抽出機で抽出した特徴を用いて生成画像と本物画像のL1距離を算出
            gen_feature = feature_extractor(gen_hr)
            real_feature = feature_extractor(imgs_hr).detach()
            loss_content = criterion_content(gen_feature, real_feature)

            # Total generator loss
            loss_G = loss_content + opt.lambda_adv * loss_GAN + opt.lambda_pixel * loss_pixel

            loss_G.backward()
            optimizer_G.step()

            optimizer_D.zero_grad()

            pred_real = discriminator(imgs_hr)
            pred_fake = discriminator(gen_hr.detach())

            # adversarial loss
            loss_real = criterion_GAN(pred_real - pred_fake.mean(0, keepdim=True), valid)
            loss_fake = criterion_GAN(pred_fake - pred_real.mean(0, keepdim=True), fake)

            loss_D = (loss_real + loss_fake) / 2

            loss_D.backward()
            optimizer_D.step()

            train_info = {
                'epoch': epoch,
                'epoch_total': opt.n_epoch,
                'batch_num': batch_num, 
                'batch_total': len(train_dataloader),
                'loss_D': loss_D.item(),
                'loss_G': loss_G.item(),
                'loss_content': loss_content.item(),
                'loss_GAN': loss_GAN.item(),
                'loss_pixel': loss_pixel.item(),
            }

            if batch_num == 1:
                sys.stdout.write('\n{}'.format(train_info))
            else:
                sys.stdout.write('\r{}'.format('\t'*20))
                sys.stdout.write('\r{}'.format(train_info))
            sys.stdout.flush()
        
        train_infos.append(train_info)
        
        if batches_done % opt.sample_interval == 0:
            # Save image grid with upsampled inputs and ESRGAN outputs
            imgs_lr = nn.functional.interpolate(imgs_lr, scale_factor=4)
            img_grid = denormalize(torch.cat((imgs_lr, gen_hr), -1))

            image_batch_save_dir = osp.join(image_train_save_dir, '{:07}'.format(batches_done))
            os.makedirs(osp.join(image_batch_save_dir, "hr_image"), exist_ok=True)
            save_image(img_grid, osp.join(image_batch_save_dir, "hr_image", "%d.png" % batches_done), nrow=1, normalize=False)

            with torch.no_grad():
                for i, imgs in enumerate(test_dataloader):
                    # Save image grid with upsampled inputs and outputs
                    imgs_lr = Variable(imgs["lr"].type(Tensor))
                    gen_hr = generator(imgs_lr)
                    imgs_lr = nn.functional.interpolate(imgs_lr, scale_factor=4)

                    imgs_lr = denormalize(imgs_lr)
                    gen_hr = denormalize(gen_hr)

                    image_batch_save_dir = osp.join(image_test_save_dir, '{:03}'.format(i))
                    os.makedirs(osp.join(image_batch_save_dir, "hr_image"), exist_ok=True)
                    save_image(gen_hr, osp.join(image_batch_save_dir, "hr_image", "{:09}.png".format(batches_done)), nrow=1, normalize=False)
                    if not low_image_save:
                        save_image(imgs_lr, osp.join(image_batch_save_dir, "lr_image.jpg"), nrow=1, normalize=False)
            low_image_save = True
                    

        if batches_done % opt.checkpoint_interval == 0:            
            # Save model checkpoints
            torch.save(generator.state_dict(), osp.join(weight_save_dir, "generator_%d.pth" % batches_done))
            torch.save(discriminator.state_dict(), osp.join(weight_save_dir, "discriminator_%d.pth" % batches_done))
        
        log_df = pd.DataFrame(train_infos)
        log_df = log_df.set_index('batch_num')
        cols = log_df.columns[log_df.columns.isin(loss_names)]
        log_df = log_df[cols]
                
        for num, loss_name in enumerate(log_df.columns, 1):
            plt.subplot(2, 3, num)
            plt.plot(log_df.index.values, log_df[loss_name].values, marker='o', color='b')
            plt.title(loss_name)
        
        display.clear_output(wait=True)
        display.display(plt.gcf())
# -










test_dataloader










loss_name









transform = transforms.Compose(
    [
        transforms.Resize()
        transforms.ToTensor(),
        transforms.Normalize(mean, std)],
)

transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std)]
)

img = Image.open('../input/cat_test/0000383.jpg')

img_size = img.size
img_width, img_height = img_size
transform = transforms.Compose([
    transforms.Resize((img_height // 4, img_width // 4), Image.BICUBIC),
    transforms.ToTensor(),
    transforms.Normalize(mean, std)])

img_trns = transform(img)

img_trns.shape

img_denorm = denormalize(img_trns)

img_prm = img_denorm.permute(1, 2, 0).numpy()

plt.imshow(img_prm)





a = img_trns.detach()

b = a.numpy()

b.transpose(0,2,1)



# +
import matplotlib.pyplot as plt
import numpy as np
import time

fig = plt.figure()
fig.show()

for j in range(10):            
    plt.plot(range(0, j), [1/(i+1) for i in range(0, j)])
    fig.canvas.draw()
    time.sleep(.05)  
# -

# %matplotlib inline
import time
import pylab as pl
from IPython import display
for i in range(10):
    pl.plot(pl.randn(100))
    display.clear_output(wait=True)
    display.display(pl.gcf())
    time.sleep(1.0)





cols = log_df.columns[log_df.columns.isin(loss_names)]







loss_names = ['batch_num', 'loss_pixel', 'loss_D', 'loss_G', 'loss_content', 'loss_GAN']

log_df = pd.DataFrame(train_infos)
log_df = log_df[loss_names]
log_df = log_df.set_index('batch_num')

# +
fig,ax = plt.subplots(2,3)

def plot_loss(log_df, ax):
    fig.figure(figsize=(16,9))
    for num, loss_name in enumerate(log_df.columns, 1):
        ax.subplot(2,3,num)
        ax.plot(log_df.index.values, log_df[loss_name].values, marker='o')
        ax.title(loss_name)
        ax.xlabel('batch_num')
# -











# +
# %matplotlib nbagg

import itertools
import math

import numpy as np
from matplotlib import pyplot as plt
from matplotlib import animation


def _update(frame, x, y):
    """グラフを更新するための関数"""
    # 現在のグラフを消去する
    plt.cla()
    # データを更新 (追加) する
    x.append(frame)
    y.append(math.sin(frame))
    # 折れ線グラフを再描画する
    plt.plot(x, y)


# 描画領域
fig = plt.figure(figsize=(10, 6))
# 描画するデータ
x = []
y = []

params = {
    'fig': fig,
    'func': _update,  # グラフを更新する関数
    'fargs': (x, y),  # 関数の引数 (フレーム番号を除く)
    'interval': 10,  # 更新間隔 (ミリ秒)
    'frames': np.arange(0, 10, 0.1),  # フレーム番号を生成するイテレータ
    'repeat': False,  # 繰り返さない
}
anime = animation.FuncAnimation(**params)
# -



# %matplotlib inline
import time
import pylab as pl
from IPython import display
for i in range(10):
    pl.plot(pl.randn(100))
    display.clear_output(wait=True)
    display.display(pl.gcf())
    time.sleep(1.0)







import pandas as pd

data:image/png;base64,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 matplotlib.pyplot as plt

plot_loss(log_df)







plot_loss











plt.figure(figsize=(16,9))
plt.subplot(2,3,1)
plt.plot(log_df.batch_num.values, log_df.loss_pixel.values, marker='o')
plt.title('loss_pixel')
plt.subplot(2,3,2)
plt.plot(log_df.batch_num.values, log_df.loss_D.values, marker='o')
plt.subplot(2,3,3)
plt.plot(log_df.batch_num.values, log_df.loss_G.values, marker='o')
plt.subplot(2,3,4)
plt.plot(log_df.batch_num.values, log_df.loss_content.values, marker='o')
plt.subplot(2,3,5)
plt.plot(log_df.batch_num.values, log_df.loss_GAN.values, marker='o')




