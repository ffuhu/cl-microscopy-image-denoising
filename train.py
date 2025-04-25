"""
@Author  : Felix Fuentes-Hurtado
@Affiliation  : National University of Singapore

Training file for the paper "CLIDiM: Contrastive Learning for Image Denoising in Microscopy"

How to run:

python train.py \
    --dataroot PAIRED_DATASET_PATH_IMAGES_IN_FORM_AB \
    --name "EXP_NAME" \
    --model pix2pix \
    --dataset_mode aligned \
    --direction AtoB \
    --crop_size 256 \
    --preprocess crop_rotate_hflip \
    --batch_size 128 \
    --num_threads 8 \
    --lr 2e-4 \
    --netG unet_256 \
    --n_epochs 500 \
    --n_epochs_decay 300 \
    --eval_only_from_epoch 150 \
    --eval_model_freq 1 \
    --save_epoch_freq 1 \
    --img_dtype uint16 \
    --cl_reg \
    --cl_reg_gen_epoch 0 \
    --lambda_tv_loss 0.0001 \
    --lambda_ssim_loss 10

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
import matplotlib.pyplot as plt


def remove_undesired_args(args_to_remove):
    print('Delete arguments for test dataloader:')
    for atr in args_to_remove:
        try:
            index = sys.argv.index(f'--{atr}')  # for testing lr does not exist, remove it
            print('Deleting:\t', sys.argv[index])
            del sys.argv[index]  # remove name
            # check if argument had a value
            if '--' not in sys.argv[index]:
                print('Deleting:\t', sys.argv[index])
                del sys.argv[index]  # remove value
        except:
            pass


if __name__ == '__main__':

    global_start_time = time.time()

    # ---------------------------------------------------------------------------------------------------------------- #
    # Setup, dataset, dataloaders, model and visualizer
    # ---------------------------------------------------------------------------------------------------------------- #

    # create train dataset
    extra_opts = {
        'seed': _SEED,
        'exp_date': datetime.datetime.now().strftime('%y%m%d_%H%M%S'),
        'pt_cudnn_benchmark': 'PT_CUDNN_BENCHMARK' in os.environ,
        'pt_detect_anomaly': 'PT_DETECT_ANOMALY' in os.environ,
    }
    opt = TrainOptions().parse(extra_opts=extra_opts)  # get training options
    dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
    dataset_size = len(dataset)  # get the number of images in the dataset.
    print('The number of training images = %d' % dataset_size)

    # create test dataset
    # careful, this is needed for test argparse
    args_to_remove = ['lr', 'n_epochs', 'n_epochs_decay', 'save_epoch_freq', 'eval_model_freq', 'continue_train',
                      'epoch_count', 'diff_data_aug', 'top_k_training', 'top_k_gamma', 'top_k_frac', 'cl_reg',
                      'cl_reg_gen_epoch', 'cl_reg_accum_iter', 'lambda_msssim_loss', 'eval_only_from_epoch',
                      'n_samples_train', 'lambda_tv_loss', 'lambda_psnr_loss', 'lambda_ssim_loss']
    remove_undesired_args(args_to_remove)
    # hard-code some parameters for test
    extra_opts = {
        'preprocess': '',
        'num_threads': 0,  # test code only supports num_threads = 0
        'batch_size': 1,  # test code only supports batch_size = 1
        # opt_test.serial_batches = True  # disable data shuffling; comment this line if results on randomly chosen images are needed.
        'no_flip': True,  # no flip; comment this line if results on flipped images are needed.
        'display_id': -1,  # no visdom display; the test code saves the results to a HTML file.
    }
    opt_test = TestOptions().parse(extra_opts=extra_opts)  # get training options
    dataset_test = create_dataset(opt_test)  # create a dataset given opt.dataset_mode and other options
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
    # F211201: add historic log for losses
    losses_history = {loss_name: [] for loss_name in model.loss_names}

    # ---------------------------------------------------------------------------------------------------------------- #
    # TRAINING LOOP
    # ---------------------------------------------------------------------------------------------------------------- #
    # outer loop for different epochs; we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>
    for epoch in range(opt.epoch_count, opt.n_epochs + opt.n_epochs_decay + 1):

        psnrs_in = []
        psnrs_pred = []
        losses_epoch = {loss_name: 0. for loss_name in model.loss_names}

        # ------------------------------------------------------------------------------------------------------------ #
        # TRAINING STAGE
        # ------------------------------------------------------------------------------------------------------------ #

        model.train()
        model.set_epoch(epoch)  # tell the model which epoch is it (useful for schedulers)

        epoch_start_time = time.time()  # timer for entire epoch
        iter_data_time = time.time()  # timer for data loading per iteration
        epoch_iter = 0  # the number of training iterations in current epoch, reset to 0 every epoch
        visualizer.reset()  # reset the visualizer: make sure it saves the results to HTML at least once every epoch
        # model.update_learning_rate()    # update learning rates in the beginning of every epoch.
        for i, data in enumerate(dataset):  # inner loop within one epoch
            iter_start_time = time.time()  # timer for computation per iteration
            if total_iters % opt.print_freq == 0:
                t_data = iter_start_time - iter_data_time

            total_iters += opt.batch_size
            epoch_iter += opt.batch_size
            model.set_input(data)  # unpack data from dataset and apply preprocessing
            model.normalize_input()
            model.optimize_parameters()  # calculate loss functions, get gradients, update network weights

            # model.denormalize_input()  # denormalize input using min and max of input images
            model.denormalize_visuals()  # denormalize visuals using min and max of input images

            if total_iters % opt.display_freq == 0 or (
                    i + 1) * opt.batch_size >= epoch_iter:  # display images on visdom and save images to a HTML file
                save_result = total_iters % opt.update_html_freq == 0
                model.compute_visuals()
                epoch_seconds_taken = int(time.time() - iter_start_time)
                # model.denormalize_visuals()
                visualizer.display_current_results(model.get_current_visuals(), epoch, epoch_seconds_taken, save_result)

            if total_iters % opt.print_freq == 0:  # print training losses and save logging information to the disk
                losses = model.get_current_losses()
                t_comp = (time.time() - iter_start_time) / opt.batch_size
                visualizer.print_current_losses(epoch, epoch_iter, losses, t_comp, t_data)
                if opt.display_id > 0:
                    visualizer.plot_current_losses(epoch, float(epoch_iter) / dataset_size, losses)

            if total_iters % opt.save_latest_freq == 0:  # cache our latest model every <save_latest_freq> iterations
                print('saving the latest model (epoch %d, total_iters %d)' % (epoch, total_iters))
                save_suffix = 'iter_%d' % total_iters if opt.save_by_iter else 'latest'
                model.save_networks(save_suffix)

            iter_data_time = time.time()

            # F211201: add historic log for losses
            losses = model.get_current_losses()
            losses_epoch = {k: v + losses[k] for k, v in losses_epoch.items()}

            # --------------------------------------------------------------------------------------------------------->
            # START - quick add to compute psnr
            # --------------------------------------------------------------------------------------------------------->

            # compute PSNR and show
            for i in range(data['A'].shape[0]):
                img_in = model.real_A[i][0].detach().cpu().numpy().astype(
                    opt.img_dtype)  # data['A'][i][0].detach().cpu().numpy()
                img_gt = model.real_B[i][0].detach().cpu().numpy().astype(
                    opt.img_dtype)  # data['B'][i][0].detach().cpu().numpy()
                img_pred = model.fake_B[i][0].detach().cpu().numpy().astype(opt.img_dtype)
                range_PSNR = np.max(img_gt) - np.min(img_gt)
                psnrs_in.append(PSNR(img_gt, img_in, max_p=range_PSNR))
                psnrs_pred.append(PSNR(img_gt, img_pred, max_p=range_PSNR))

        psnrs_in_mean = np.mean(psnrs_in)
        psnrs_pred_mean = np.mean(psnrs_pred)
        psnr_txt = '[%d]\t[TRAIN]\tPSNR_in_mean: %.2f\tPSNR_pred_mean: %.2f' % (epoch, psnrs_in_mean, psnrs_pred_mean)
        print(psnr_txt)
        with open(visualizer.log_name, "a") as log_file:
            log_file.write('%s\n' % psnr_txt)
        # --------------------------------------------------------------------------------------------------------->
        # END - quick add to compute psnr
        # --------------------------------------------------------------------------------------------------------->

        # if epoch % opt.save_epoch_freq == 0:  # cache our model every <save_epoch_freq> epochs
        #     print('saving the model at the end of epoch %d, iters %d' % (epoch, total_iters))
        #     model.save_networks('latest')
        #     model.save_networks(epoch)

        print('End of epoch %d / %d \t Time Taken: %d sec' % (
            epoch, opt.n_epochs + opt.n_epochs_decay, time.time() - epoch_start_time))
        model.update_learning_rate()  # update learning rates in the beginning of every epoch.
        # F210820: log learning rate updates to the log file
        visualizer.log_learning_rate_updates(epoch, model)

        # F211201: add historic log for losses
        losses_history = {k: [*v, losses_epoch[k]] for k, v in losses_history.items()}

        # ------------------------------------------------------------------------------------------------------------ #
        # TEST STAGE
        # ------------------------------------------------------------------------------------------------------------ #
        if epoch == 0 or epoch >= opt.eval_only_from_epoch and epoch % opt.eval_model_freq == 0:
            model.eval()

            psnrs_in = []
            psnrs_pred = []

            metrics = []
            # metrics_tiled = []

            # create a website
            # test with eval mode. This only affects layers like batchnorm and dropout.
            # For [pix2pix]: we use batchnorm and dropout in the original pix2pix.
            # You can experiment it with and without eval() mode.
            # For [CycleGAN]: It should not affect CycleGAN as CycleGAN uses instancenorm without dropout.
            heading = 'Experiment = %s, Phase = %s, Epoch = %s' % (opt.name, opt.phase, opt.epoch)
            webpage = html.HTML(visualizer.web_dir, heading)
            for i, data in enumerate(tqdm(dataset_test.dataloader, desc='Predicting test set')):
                if i >= opt_test.num_test:  # only apply our model to opt.num_test images.
                    break
                model.set_input(data)  # unpack data from data loader
                model.normalize_input()  # normalize input using min and max of input images
                model.test()  # run inference

                # model.denormalize_input()  # denormalize input using min and max of input images
                model.denormalize_visuals()  # denormalize visuals using min and max of input images

                visuals = model.get_current_visuals()  # get image results
                img_path = model.get_image_paths()  # get image paths
                # if i % opt_test.test_model_freq == 0:  # save images to an HTML file
                #     print('processing (%04d)-th image... %s' % (i, img_path))
                m_ = compute_metrics_and_save_images(webpage, visuals, img_path,
                                                     aspect_ratio=opt_test.aspect_ratio,
                                                     width=opt_test.display_winsize,
                                                     dtype=opt.img_dtype)
                metrics.extend(m_)

            # compute average and std of metrics and save images to html and metrics to html and csv
            psnrs_in_mean, psnrs_pred_mean = compute_moments_and_save_to_html(webpage, metrics)
            save_metrics_to_csv(os.path.join(visualizer.web_dir, f'metrics_{opt.name}.csv'), metrics, epoch)

            # --------------------------------------------------------------------------------------------------------->
            # quick add to compute psnr
            # --------------------------------------------------------------------------------------------------------->
            psnr_txt = '[%d]\t[TEST]\tPSNR_in_mean: %.2f\tPSNR_pred_mean: %.2f' % (
                epoch, psnrs_in_mean, psnrs_pred_mean)
            print(psnr_txt)
            with open(visualizer.log_name, "a") as log_file:
                log_file.write('%s\n' % psnr_txt)
            # --------------------------------------------------------------------------------------------------------->

            webpage.save('test_%i.html' % epoch)  # save the HTML

            # save the model only if PSNRpred > PSNRin and PSNRpred > PSNRpred_last_best
            if epoch == 0 or psnrs_pred_mean > psnrs_in_mean and psnrs_pred_mean > last_best_test_PSNR:
                print('saving the model at the end of epoch %d (PSNR_pred=%.2f)' % (epoch, psnrs_pred_mean))
                model.save_networks(epoch)
                saved_models.append([epoch, psnrs_pred_mean])
                last_best_test_PSNR = psnrs_pred_mean

                # clean saved models - keep only 5 best
                models_exceed = len(saved_models) - opt.n_best_models_to_keep
                if models_exceed > 0:
                    for i in range(models_exceed):
                        model_epoch = saved_models[i][0]

                        # delete the checkpoints
                        try:
                            for name in model.model_names:
                                save_filename = '%s_net_%s.pth' % (model_epoch, name)
                                model_path = os.path.join(model.save_dir, save_filename)
                                msg = 'Remove: ' + model_path
                                print(msg)
                                with open(visualizer.log_name, "a") as log_file:
                                    log_file.write('%s\n' % msg)
                                os.remove(model_path)
                                del saved_models[i]
                        except:
                            msg = 'ERROR removing: ' + model_path
                            print(msg)
                            with open(visualizer.log_name, "a") as log_file:
                                log_file.write('%s\n' % msg)

    # F211201: add historic log for losses
    for loss_name, loss_values in losses_history.items():
        plt.plot(range(len(loss_values)), loss_values)
    plt.legend(list(losses_history.keys()))
    plt.savefig(model.save_dir + os.sep + f'plot_losses.png')

    print(f'Total time taken: {time.time() - global_start_time} sec.')
