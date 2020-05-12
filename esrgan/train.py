import argparse
import os
import os.path as osp
import time
import datetime

import numpy as np
import sys

from torch.utils.data import DataLoader
from torch.autograd import Variable

from models import GeneratorRRDB, Discriminator, FeatureExtractor
from datasets import ImageDataset, denormalize

import torch.nn as nn
import torch
from torchvision.utils import save_image

import mlflow

ROOT = './'
mlflow.set_tracking_uri(osp.join(ROOT, "mlruns"))
mlflow.set_experiment('esrgan')

input_dir = osp.join(ROOT, 'input')

output_dir = osp.join(ROOT, 'output', str(datetime.datetime.fromtimestamp(time.time())))
os.makedirs(output_dir, exist_ok=True)

image_train_save_dir = osp.join(output_dir, 'image', 'train')
image_valid_save_dir = osp.join(output_dir, 'image', 'valid')
image_test_save_dir = osp.join(output_dir, 'image', 'test')
weight_save_dir = osp.join(output_dir, 'weight')

os.makedirs(image_train_save_dir, exist_ok=True)
os.makedirs(image_test_save_dir, exist_ok=True)
os.makedirs(weight_save_dir, exist_ok=True)


def build_argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epoch", type=int, default=0, help="epoch to start training from")
    parser.add_argument("--n_epochs", type=int, default=200, help="number of epochs of training")
    parser.add_argument("--dataset_name", type=str, default="img_align_celeba_resize", help="name of the dataset")
    parser.add_argument("--batch_size", type=int, default=4, help="size of the batches")
    parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
    parser.add_argument("--b1", type=float, default=0.9, help="adam: decay of first order momentum of gradient")
    parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
    parser.add_argument("--decay_epoch", type=int, default=100, help="epoch from which to start lr decay")
    parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
    parser.add_argument("--hr_height", type=int, default=160, help="high res. image height")
    parser.add_argument("--hr_width", type=int, default=160, help="high res. image width")
    parser.add_argument("--channels", type=int, default=3, help="number of image channels")
    parser.add_argument("--sample_interval", type=int, default=100, help="interval between saving image samples")
    parser.add_argument("--checkpoint_interval", type=int, default=5000, help="batch interval between model checkpoints")
    parser.add_argument("--residual_blocks", type=int, default=23, help="number of residual blocks in the generator")
    parser.add_argument("--warmup_batches", type=int, default=500, help="number of batches with pixel-wise loss only")
    parser.add_argument("--lambda_adv", type=float, default=5e-3, help="adversarial loss weight")
    parser.add_argument("--lambda_pixel", type=float, default=1e-2, help="pixel-wise loss weight")
    opt = parser.parse_args()
    print(opt)
    return opt


def main(opt):
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

    for key, value in vars(opt).items():
        mlflow.log_param(key, value)

    if opt.epoch != 0:
        # Load pretrained models
        load_g_weight_path = osp.join(weight_save_dir, "generator_%d.pth" % opt.epoch)
        load_d_weight_path = osp.join(weight_save_dir, "discriminator_%d.pth" % opt.epoch)

        generator.load_state_dict(torch.load(load_g_weight_path))
        discriminator.load_state_dict(torch.load(load_d_weight_path))

    # Optimizers
    optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

    Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.Tensor

    # dataloader
    train_data_path = osp.join(input_dir, '{}_train'.format(opt.dataset_name))
    print(train_data_path)
    train_dataloader = DataLoader(
        ImageDataset(train_data_path, hr_shape=hr_shape),
        batch_size=opt.batch_size,
        shuffle=True,
        num_workers=opt.n_cpu,
    )

    valid_dataloader = DataLoader(
        ImageDataset(osp.join(input_dir, '{}_valid'.format(opt.dataset_name)), hr_shape=hr_shape),
        batch_size=opt.batch_size,
        shuffle=False,
        num_workers=opt.n_cpu, 
    )

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

                mlflow.log_metric('train_{}'.format('loss_pixel'), loss_pixel.item(), step=batches_done)
                if batches_done % opt.sample_interval == 0:
                    with torch.no_grad():
                        for i, imgs in enumerate(valid_dataloader):
                            # Save image grid with upsampled inputs and outputs
                            imgs_lr = Variable(imgs["lr"].type(Tensor))
                            gen_hr = generator(imgs_lr)
                            imgs_lr = nn.functional.interpolate(imgs_lr, scale_factor=4)
                            
                            imgs_lr = denormalize(imgs_lr)
                            gen_hr = denormalize(gen_hr)

                            image_batch_save_dir = osp.join(image_valid_save_dir, '{:07}'.format(batches_done))
                            os.makedirs(osp.join(image_batch_save_dir, "lr_image"), exist_ok=True)
                            os.makedirs(osp.join(image_batch_save_dir, "hr_image"), exist_ok=True)

                            save_image(imgs_lr, osp.join(image_batch_save_dir, "lr_image", "{:09}.png".format(i)), nrow=1, normalize=False)
                            save_image(gen_hr, osp.join(image_batch_save_dir, "hr_image", "{:09}.png".format(i)), nrow=1, normalize=False)

                            # # if i == 1:
                            # #     sys.stdout.write("\n{}/{}".format(i, len(test_dataloader)))
                            # # else:
                            # #     sys.stdout.write("\r{}/{}".format(i, len(test_dataloader)))

                            # sys.stdout.flush()

                            if i > 5:
                                break

                continue
            
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

            # import pdb; pdb.set_trace()

            if batches_done % opt.sample_interval == 0:
                # Save image grid with upsampled inputs and ESRGAN outputs
                imgs_lr = nn.functional.interpolate(imgs_lr, scale_factor=4)
                img_grid = denormalize(torch.cat((imgs_lr, gen_hr), -1))

                image_batch_save_dir = osp.join(image_train_save_dir, '{:07}'.format(batches_done))
                os.makedirs(osp.join(image_batch_save_dir, "hr_image"), exist_ok=True)
                save_image(img_grid, osp.join(image_batch_save_dir, "hr_image", "%d.png" % batches_done), nrow=1, normalize=False)

            # # if batches_done % opt.sample_interval == 0:
            # if batches_done % opt.sample_interval == 0:
                with torch.no_grad():
                    for i, imgs in enumerate(valid_dataloader):
                        # Save image grid with upsampled inputs and outputs
                        imgs_lr = Variable(imgs["lr"].type(Tensor))
                        gen_hr = generator(imgs_lr)
                        imgs_lr = nn.functional.interpolate(imgs_lr, scale_factor=4)
                        
                        imgs_lr = denormalize(imgs_lr)
                        gen_hr = denormalize(gen_hr)

                        image_batch_save_dir = osp.join(image_valid_save_dir, '{:07}'.format(batches_done))
                        os.makedirs(osp.join(image_batch_save_dir, "lr_image"), exist_ok=True)
                        os.makedirs(osp.join(image_batch_save_dir, "hr_image"), exist_ok=True)

                        save_image(imgs_lr, osp.join(image_batch_save_dir, "lr_image", "{:09}.png".format(i)), nrow=1, normalize=False)
                        save_image(gen_hr, osp.join(image_batch_save_dir, "hr_image", "{:09}.png".format(i)), nrow=1, normalize=False)

                        # # if i == 1:
                        # #     sys.stdout.write("\n{}/{}".format(i, len(test_dataloader)))
                        # # else:
                        # #     sys.stdout.write("\r{}/{}".format(i, len(test_dataloader)))

                        # sys.stdout.flush()

                        if i > 5:
                            break

            if batches_done % opt.checkpoint_interval == 0:            
                # Save model checkpoints
                torch.save(generator.state_dict(), osp.join(weight_save_dir, "generator_%d.pth" % epoch))
                torch.save(discriminator.state_dict(), osp.join(weight_save_dir, "discriminator_%d.pth" % epoch))

            mlflow.log_metric('train_{}'.format('loss_D'), loss_D.item(), step=batches_done)
            mlflow.log_metric('train_{}'.format('loss_G'), loss_G.item(), step=batches_done)
            mlflow.log_metric('train_{}'.format('loss_content'), loss_content.item(), step=batches_done)
            mlflow.log_metric('train_{}'.format('loss_GAN'), loss_GAN.item(), step=batches_done)
            mlflow.log_metric('train_{}'.format('loss_pixel'), loss_pixel.item(), step=batches_done)

if __name__ == '__main__':
    opt = build_argparser()
    sys.exit(main(opt) or 0)
