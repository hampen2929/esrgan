# -*- coding: utf-8 -*-
# # import 

import argparse
import os
import os.path as osp
import numpy as np
import sys
from tqdm import tqdm

from torch.utils.data import DataLoader
from torch.autograd import Variable
import torch.nn as nn
import torch
from torchvision.utils import save_image

from models import GeneratorRRDB, Discriminator, FeatureExtractor
from datasets import DemoImageDataset, denormalize


# # Option

class Opt():
    def __init__(self):
        self.channles = 3
        self.hr_height = 128
        self.residual_blocks = 23
        self.lr = 0.0002
        self.b1 = 0.9
        self.b2 = 0.999
        self.batch_size = 1
        self.n_cpu = 8
        self.n_epoch = 200
        # opt.warmup_batches = 500
        self.warmup_batches = 5
        self.lambda_adv = 5e-3
        self.lambda_pixel = 1e-2
        self.pretrained = False
        self.dataset_name = 'cat'

        self.sample_interval = 50
        self.checkpoint_interval = 100
        
        self.hr_height = 128
        self.hr_width = 128
        self.channels = 3
        
        self.epoch = 0
        self.n_epochs = 200


opt = Opt()

# # path

ROOT = '../'

dataset_name = 'cat_face'

# +
demo_in_dir = osp.join(ROOT, 'input', dataset_name, 'demo')
demo_out_dir = osp.join(ROOT, 'output', dataset_name)

demo_in_dir = '/workspace/demo/input'
demo_out_dir = '/workspace/demo/output'
# -

os.makedirs(demo_out_dir, exist_ok=True)

weight_path = "/workspace/output/cat_face/weight/generator_3900.pth"

# # data

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

hr_shape = (opt.hr_height, opt.hr_width)

# +
# Initialize generator and discriminator
generator = GeneratorRRDB(opt.channels, filters=64, num_res_blocks=opt.residual_blocks).to(device)
generator.load_state_dict(torch.load(weight_path))

Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.Tensor
# -

demo_dataloader = DataLoader(
    DemoImageDataset(demo_in_dir),
    batch_size=opt.batch_size,
    shuffle=False,
    num_workers=opt.n_cpu,
)


# # generate hr image


# 入力された画像を高解像度画像、リサイズで縦横を1/4にした画像を低解像度画像として、低解像度画像から高解像度画像を生成する

with torch.no_grad():
    for i, imgs in enumerate(demo_dataloader):
        # Save image grid with upsampled inputs and outputs
        imgs_lr = Variable(imgs["lr"].type(Tensor))
        gen_hr = generator(imgs_lr)
        imgs_lr = nn.functional.interpolate(imgs_lr, scale_factor=4)
        
        imgs_lr = denormalize(imgs_lr)
        gen_hr = denormalize(gen_hr)

        os.makedirs(demo_out_dir, exist_ok=True)
                
        save_image(imgs_lr, osp.join(demo_out_dir, "low_{:01}.png".format(i)), nrow=1, normalize=False)
        save_image(gen_hr, osp.join(demo_out_dir, "gen_hr_{:01}.png".format(i)), nrow=1, normalize=False)


