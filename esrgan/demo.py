import argparse
import os
import os.path as osp
import numpy as np
import sys
from tqdm import tqdm

from torch.utils.data import DataLoader
from torch.autograd import Variable

from models import GeneratorRRDB, Discriminator, FeatureExtractor
from datasets import ImageDataset, denormalize

import torch.nn as nn
import torch
from torchvision.utils import save_image

import mlflow

mlflow.set_tracking_uri("../mlruns")
mlflow.set_experiment('esrgan')

ROOT = '../'

demo_in_dir = osp.join(ROOT, 'demo', 'input')
demo_out_dir = osp.join(ROOT, 'demo', 'output')

os.makedirs(demo_out_dir, exist_ok=True)

parser = argparse.ArgumentParser()
parser.add_argument("--epoch", type=int, default=1, help="epoch to start training from")
parser.add_argument("--n_epochs", type=int, default=200, help="number of epochs of training")
parser.add_argument("--dataset_name", type=str, default="img_align_celeba", help="name of the dataset")
parser.add_argument("--batch_size", type=int, default=4, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.9, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--decay_epoch", type=int, default=100, help="epoch from which to start lr decay")
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
parser.add_argument("--hr_height", type=int, default=256, help="high res. image height")
parser.add_argument("--hr_width", type=int, default=256, help="high res. image width")
parser.add_argument("--channels", type=int, default=3, help="number of image channels")
parser.add_argument("--sample_interval", type=int, default=100, help="interval between saving image samples")
parser.add_argument("--checkpoint_interval", type=int, default=5000, help="batch interval between model checkpoints")
parser.add_argument("--residual_blocks", type=int, default=23, help="number of residual blocks in the generator")
parser.add_argument("--warmup_batches", type=int, default=500, help="number of batches with pixel-wise loss only")
parser.add_argument("--lambda_adv", type=float, default=5e-3, help="adversarial loss weight")
parser.add_argument("--lambda_pixel", type=float, default=1e-2, help="pixel-wise loss weight")
opt = parser.parse_args()
print(opt)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

hr_shape = (opt.hr_height, opt.hr_width)

# Initialize generator and discriminator
generator = GeneratorRRDB(opt.channels, filters=64, num_res_blocks=opt.residual_blocks).to(device)
generator.load_state_dict(torch.load("saved_models/generator_%d.pth" % opt.epoch))
Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.Tensor

demo_dataloader = DataLoader(
    ImageDataset(demo_in_dir, hr_shape=hr_shape),
    batch_size=opt.batch_size,
    shuffle=False,
    num_workers=opt.n_cpu,
)


# ----------
#  Training
# ----------


with torch.no_grad():
    for i, imgs in enumerate(demo_dataloader):
        # Save image grid with upsampled inputs and outputs
        imgs_lr = Variable(imgs["lr"].type(Tensor))
        gen_hr = generator(imgs_lr)
        imgs_lr = nn.functional.interpolate(imgs_lr, scale_factor=4)
        
        imgs_lr = denormalize(imgs_lr)
        gen_hr = denormalize(gen_hr)

        image_test_batch_save_dir = osp.join(demo_out_dir, '{:03}'.format(i))
        os.makedirs(osp.join(image_test_batch_save_dir, "low"), exist_ok=True)
        os.makedirs(osp.join(image_test_batch_save_dir, "high"), exist_ok=True)

        save_image(imgs_lr, osp.join(image_test_batch_save_dir, "low", "{:09}.png".format(i)), nrow=1, normalize=False)
        save_image(gen_hr, osp.join(image_test_batch_save_dir, "high", "{:09}.png".format(i)), nrow=1, normalize=False)

        # # if i == 1:
        # #     sys.stdout.write("\n{}/{}".format(i, len(test_dataloader)))
        # # else:
        # #     sys.stdout.write("\r{}/{}".format(i, len(test_dataloader)))

        # sys.stdout.flush()

        if i > 5:
            break
