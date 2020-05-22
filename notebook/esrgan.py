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
plot_save_dir = osp.join(output_dir, 'plot')

save_dirs = [image_train_save_dir, image_test_save_dir, weight_save_dir, plot_save_dir]
for save_dir in save_dirs:
    os.makedirs(save_dir, exist_ok=True)
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
                plt.plot(log_df.index.values, log_df[loss_name].values, marker='o', color='b', alpha=0.8)
                plt.title(loss_name)

            plt.savefig(osp.join(plot_save_dir, "plot.png"))

            # display.clear_output(wait=True)
            # display.display(plt.gcf())
# -



