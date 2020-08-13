# # import 

# +
import argparse
import os
import os.path as osp
import time
import datetime

import numpy as np
import sys

from torch.utils.data import DataLoader
from torch.autograd import Variable
import torch.nn as nn
import torch
from torchvision.utils import save_image
# -

from models import GeneratorRRDB, Discriminator, FeatureExtractor
from datasets import ImageDataset, denormalize

# # constang

dataset_name = 'cat_face'
ROOT = '../'

# # path

input_dir = osp.join(ROOT, 'input')

output_dir = osp.join(ROOT, 'output', dataset_name)
# output_dir = osp.join(ROOT, 'output', str(datetime.datetime.fromtimestamp(time.time())))
os.makedirs(output_dir, exist_ok=True)

train_data_path = osp.join(input_dir, dataset_name, 'train')
test_data_path = osp.join(input_dir, dataset_name, 'test')

image_train_save_dir = osp.join(output_dir, 'train')
# image_valid_save_dir = osp.join(output_dir, 'image', 'valid')
image_test_save_dir = osp.join(output_dir, 'test')
weight_save_dir = osp.join(output_dir, 'weight')

os.makedirs(image_train_save_dir, exist_ok=True)
os.makedirs(image_test_save_dir, exist_ok=True)
os.makedirs(weight_save_dir, exist_ok=True)


class Opt():
    def __init__(self):
        self.channles = 3
        self.hr_height = 128
        self.residual_blocks = 23
        self.lr = 0.0002
        self.b1 = 0.9
        self.b2 = 0.999
        self.batch_size = 4
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


# # main

# +
# def main(opt):
# -

opt = Opt()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
hr_shape = (opt.hr_height, opt.hr_width)

# Initialize generator and discriminator
generator = GeneratorRRDB(opt.channels, filters=64, num_res_blocks=opt.residual_blocks).to(device)
discriminator = Discriminator(input_shape=(opt.channels, *hr_shape)).to(device)
feature_extractor = FeatureExtractor().to(device)

# Set feature extractor to inference mode
feature_extractor.eval()

# Losses
criterion_GAN = torch.nn.BCEWithLogitsLoss().to(device)
criterion_content = torch.nn.L1Loss().to(device)
criterion_pixel = torch.nn.L1Loss().to(device)

# Optimizers
optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.Tensor

# +
# dataloader
train_dataloader = DataLoader(
    ImageDataset(train_data_path, hr_shape=hr_shape),
    batch_size=opt.batch_size,
    shuffle=True,
    num_workers=opt.n_cpu,
)

test_dataloader = DataLoader(
    ImageDataset(test_data_path, hr_shape=hr_shape),
    batch_size=opt.batch_size,
    shuffle=False,
    num_workers=opt.n_cpu,
)

# +
# ----------
#  Training
# ----------

for epoch in range(opt.epoch + 1, opt.n_epochs + 1):
    for batch_num, imgs in enumerate(train_dataloader):
        batches_done = (epoch - 1) * len(train_dataloader) + batch_num

        # Configure model input
        imgs_lr = Variable(imgs["lr"].type(Tensor))
        imgs_hr = Variable(imgs["hr"].type(Tensor))

        # Adversarial ground truths
        valid = Variable(Tensor(np.ones((imgs_lr.size(0), *discriminator.output_shape))), requires_grad=False)
        fake = Variable(Tensor(np.zeros((imgs_lr.size(0), *discriminator.output_shape))), requires_grad=False)

        # ------------------
        #  Train Generators
        # ------------------

        optimizer_G.zero_grad()

        # Generate a high resolution image from low resolution input
        gen_hr = generator(imgs_lr)

        # Measure pixel-wise loss against ground truth
        loss_pixel = criterion_pixel(gen_hr, imgs_hr)

        if batches_done <= opt.warmup_batches:
            # Warm-up (pixel-wise loss only)
            loss_pixel.backward()
            optimizer_G.step()
            log_info = "[Epoch {}/{}] [Batch {}/{}] [G pixel: {}]".format(epoch, opt.n_epochs, batch_num, len(train_dataloader), loss_pixel.item())

            sys.stdout.write("\r{}".format(log_info))
            sys.stdout.flush()
        else:
            # Extract validity predictions from discriminator
            pred_real = discriminator(imgs_hr).detach()
            pred_fake = discriminator(gen_hr)

            # Adversarial loss (relativistic average GAN)
            loss_GAN = criterion_GAN(pred_fake - pred_real.mean(0, keepdim=True), valid)

            # Content loss
            gen_features = feature_extractor(gen_hr)
            real_features = feature_extractor(imgs_hr).detach()
            loss_content = criterion_content(gen_features, real_features)

            # Total generator loss
            loss_G = loss_content + opt.lambda_adv * loss_GAN + opt.lambda_pixel * loss_pixel

            loss_G.backward()
            optimizer_G.step()

            # ---------------------
            #  Train Discriminator
            # ---------------------

            optimizer_D.zero_grad()

            pred_real = discriminator(imgs_hr)
            pred_fake = discriminator(gen_hr.detach())

            # Adversarial loss for real and fake images (relativistic average GAN)
            loss_real = criterion_GAN(pred_real - pred_fake.mean(0, keepdim=True), valid)
            loss_fake = criterion_GAN(pred_fake - pred_real.mean(0, keepdim=True), fake)

            # Total loss
            loss_D = (loss_real + loss_fake) / 2

            loss_D.backward()
            optimizer_D.step()

            # --------------
            #  Log Progress
            # --------------                    

            log_info = "[Epoch {}/{}] [Batch {}/{}] [D loss: {}] [G loss: {}, content: {}, adv: {}, pixel: {}]".format(
                    epoch,
                    opt.n_epochs,
                    batch_num,
                    len(train_dataloader),
                    loss_D.item(),
                    loss_G.item(),
                    loss_content.item(),
                    loss_GAN.item(),
                    loss_pixel.item(),
                )

            if batch_num == 1:
                sys.stdout.write("\n{}".format(log_info))
            else:
                sys.stdout.write("\r{}".format(log_info))

            sys.stdout.flush()

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

                    image_batch_save_dir = osp.join(image_test_save_dir, '{:07}'.format(batches_done))
                    os.makedirs(osp.join(image_batch_save_dir, "lr_image"), exist_ok=True)
                    os.makedirs(osp.join(image_batch_save_dir, "hr_image"), exist_ok=True)

                    save_image(imgs_lr, osp.join(image_batch_save_dir, "lr_image", "{:09}.png".format(i)), nrow=1, normalize=False)
                    save_image(gen_hr, osp.join(image_batch_save_dir, "hr_image", "{:09}.png".format(i)), nrow=1, normalize=False)

                    if i > 25:
                        break

        if batches_done % opt.checkpoint_interval == 0:            
            # Save model checkpoints
            torch.save(generator.state_dict(), osp.join(weight_save_dir, "generator_%d.pth" % batches_done))
            torch.save(discriminator.state_dict(), osp.join(weight_save_dir, "discriminator_%d.pth" % batches_done))


# +
# if __name__ == '__main__':
#     opt = Opt()
#     sys.exit(main(opt) or 0)
# -


