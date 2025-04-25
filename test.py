"""
@Author  : Felix Fuentes-Hurtado
@Affiliation  : National University of Singapore

Test file for the paper "CLIDiM: Contrastive Learning for Image Denoising in Microscopy"

How to run:

python test.py \
    --dataroot PAIRED_DATASET_PATH_IMAGES_IN_FORM_AB \
    --name "TRAINED_EXP_PATH" \
    --model pix2pix \
    --load_iter '123' \
    --dataset_mode aligned \
    --direction AtoB \
    --netG unet_256
    

"""
import os
import random
import numpy as np
import torch

# for reproducibility
_SEED = int(os.environ['SEED']) if 'SEED' in os.environ else 42
random.seed(_SEED)
np.random.seed(_SEED)
torch.manual_seed(_SEED)
if 'PT_CUDNN_BENCHMARK' in os.environ:
    torch.backends.cudnn.benchmark = False
    print('Using "torch.backends.cudnn.benchmark = False"')
if 'PT_DETECT_ANOMALY' in os.environ:
    torch.autograd.set_detect_anomaly(True)
    print('Using "torch.autograd.set_detect_anomaly(True)"')

import time
import datetime
import sys
from options.train_options import TrainOptions
from options.test_options import TestOptions
from data import create_dataset
from models import create_model
from util import util, html
from util.visualizer import Visualizer
from util.visualizer import compute_metrics_and_save_images, compute_moments_and_save_to_html, save_metrics_to_csv

from tqdm import tqdm
from util.metrics import PSNR

if __name__ == '__main__':

    # ---------------------------------------------------------------------------------------------------------------- #
    # Setup, dataset, dataloaders, model and visualizer
    # ---------------------------------------------------------------------------------------------------------------- #
    # hard-code some parameters for test
    extra_opts = {
        'preprocess': '',
        'num_threads': 0,  # test code only supports num_threads = 0
        'batch_size': 1,  # test code only supports batch_size = 1
        # opt_test.serial_batches = True  # disable data shuffling; comment this line if results on randomly chosen images are needed.
        'no_flip': True,  # no flip; comment this line if results on flipped images are needed.
        'display_id': -1,  # no visdom display; the test code saves the results to a HTML file.
        'display_port': 8097,  # no visdom display; the test code saves the results to a HTML file.
    }
    opt = TestOptions().parse(extra_opts=extra_opts)  # get training options
    dataset_test = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
    dataset_test_size = len(dataset_test)  # get the number of images in the dataset.
    print('The number of test images = %d' % dataset_test_size)

    # WARNING: model with only be evaluated during epoch_decay epochs

    model = create_model(opt)  # create a model given opt.model and other options
    model.setup(opt)  # regular setup: load and print networks; create schedulers

    visualizer = Visualizer(opt)  # create a visualizer that display/save images and plots
    total_iters = 0  # the total number of training iterations

    # to keep track of saved models
    psnrs_in_mean = 0
    psnrs_pred_mean = 0
    last_best_test_PSNR = 0
    saved_models = []

    # ------------------------------------------------------------------------------------------------------------ #
    # TEST STAGE
    # ------------------------------------------------------------------------------------------------------------ #

    model.eval()

    psnrs_in = []
    psnrs_pred = []
    metrics = []

    # create a website
    # test with eval mode. This only affects layers like batchnorm and dropout.
    # For [pix2pix]: we use batchnorm and dropout in the original pix2pix.
    # You can experiment it with and without eval() mode.
    # For [CycleGAN]: It should not affect CycleGAN as CycleGAN uses instancenorm without dropout.
    heading = 'Experiment = %s, Phase = %s, Epoch = %s' % (opt.name, opt.phase, opt.epoch)
    webpage = html.HTML(visualizer.web_dir, heading)
    for i, data in enumerate(tqdm(dataset_test.dataloader, desc='Predicting test set')):
        model.set_input(data)  # unpack data from data loader
        model.normalize_input()  # normalize input using min and max of input images
        model.test()  # run inference

        model.denormalize_visuals()  # denormalize visuals using min and max of input images

        visuals = model.get_current_visuals()  # get image results
        img_path = model.get_image_paths()  # get image paths

        m_ = compute_metrics_and_save_images(webpage, visuals, img_path,
                                             aspect_ratio=opt.aspect_ratio,
                                             width=opt.display_winsize,
                                             dtype=opt.img_dtype)
        metrics.extend(m_)

    # compute average and std of metrics and save images to html and metrics to html and csv
    psnrs_in_mean, psnrs_pred_mean = compute_moments_and_save_to_html(webpage, metrics)
    print(metrics)
    # save_metrics_to_csv(os.path.join(visualizer.web_dir, f'metrics_{opt.name}.csv'), metrics, args.test_name)

    # --------------------------------------------------------------------------------------------------------->
    # compute psnr
    # --------------------------------------------------------------------------------------------------------->
    psnr_txt = '[TEST]\tPSNR_in_mean: %.2f\tPSNR_pred_mean: %.2f' % (psnrs_in_mean, psnrs_pred_mean)
    print(psnr_txt)

    # with open(visualizer.log_name, "a") as log_file:
    #     log_file.write('%s\n' % psnr_txt)
    # # --------------------------------------------------------------------------------------------------------->
    #
    # webpage.save('test_%s.html' % opt.name)  # save the HTML
