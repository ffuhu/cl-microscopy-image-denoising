import torch
import math
from .base_model import BaseModel
from . import networks
from data.diff_aug import DiffAugment  # F210729: add "Differentiable Augmentation for Data-Efficient GAN Training"


class Pix2PixModel(BaseModel):
    """ This class implements the pix2pix model, for learning a mapping from input images to output images given paired data.

    The model training requires '--dataset_mode aligned' dataset.
    By default, it uses a '--netG unet256' U-Net generator,
    a '--netD basic' discriminator (PatchGAN),
    and a '--gan_mode' vanilla GAN loss (the cross-entropy objective used in the orignal GAN paper).

    pix2pix paper: https://arxiv.org/pdf/1611.07004.pdf
    """

    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """Add new dataset-specific options, and rewrite default values for existing options.

        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.

        For pix2pix, we do not use image buffer
        The training objective is: GAN Loss + lambda_L1 * ||G(A)-B||_1
        By default, we use vanilla GAN loss, UNet with batchnorm, and aligned datasets.
        """
        # changing the default values to match the pix2pix paper (https://phillipi.github.io/pix2pix/)
        parser.set_defaults(norm='batch', netG='unet_256', dataset_mode='aligned')
        if is_train:
            parser.set_defaults(pool_size=0, gan_mode='vanilla')
            parser.add_argument('--lambda_L1', type=float, default=100.0, help='weight for L1 loss')

        return parser

    def __init__(self, opt):
        """Initialize the pix2pix class.

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseModel.__init__(self, opt)
        # specify the training losses you want to print out. The training/test scripts will call <BaseModel.get_current_losses>
        self.loss_names = ['G', 'G_GAN', 'G_L1', 'D_real', 'D_fake']
        # specify the images you want to save/display. The training/test scripts will call <BaseModel.get_current_visuals>
        self.visual_names = ['real_A', 'fake_B', 'real_B']
        # specify the models you want to save to the disk. The training/test scripts will call <BaseModel.save_networks> and <BaseModel.load_networks>
        if self.isTrain:
            self.model_names = ['G', 'D']
        else:  # during test time, only load G
            self.model_names = ['G']
        # define networks (both generator and discriminator)
        self.netG = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.norm, opt.upsampling,
                                      not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)

        if self.isTrain:

            # F210929: add losses only if they are used
            if self.opt.cl_reg:
                self.loss_names.append('D_cl')
            if self.opt.lambda_msssim_loss:
                self.loss_names.append('G_MSSSIM')
            if self.opt.lambda_ssim_loss:
                self.loss_names.append('G_SSIM')
            if self.opt.lambda_tv_loss:
                self.loss_names.append('G_TV')
            if self.opt.lambda_psnr_loss:
                self.loss_names.append('G_PSNR')

            # define a discriminator; conditional GANs need to take both input and output images; Therefore, #channels for D is input_nc + output_nc
            self.netD = networks.define_D(opt.input_nc + opt.output_nc, opt.ndf, opt.netD,
                                          opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)

            # define loss functions
            self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)
            self.criterionL1 = torch.nn.L1Loss()
            # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)

            self.data_range = 2 ** 16 - 1 if self.opt.img_dtype == 'uint16' else 255

            # F210729: add "Differentiable Augmentation for Data-Efficient GAN Training"
            # specify the smooth data augmentation, default is none, options include translation, change on color and random cutout
            self.diff_data_aug = opt.diff_data_aug

            # F210729: add Top-k GAN Training
            self.epoch = opt.epoch_count
            self.iter = opt.iter_count
            self.batch_size = opt.batch_size
            self.top_k_training = opt.top_k_training
            self.top_k_gamma = opt.top_k_gamma
            self.top_k_frac = opt.top_k_frac

            # F210818: add MSSIM+L1 (Mix) loss
            # https://github.com/NVlabs/PL4NN
            # https://github.com/VainF/pytorch-msssim
            self.lambda_msssim_loss = self.opt.lambda_msssim_loss
            if self.lambda_msssim_loss > 0:
                from pytorch_msssim import MS_SSIM
                self.criterionMSSSIM = MS_SSIM(data_range=1., size_average=True, channel=self.opt.output_nc)

            # F210929: add Contrastive-Learning regularization
            self.cl_reg = opt.cl_reg
            self.cl_reg_gen_epoch = opt.cl_reg_gen_epoch
            # self.cl_reg_accum_iter = opt.cl_reg_accum_iter
            self.cl_reg_apply_to_generated = False
            if self.cl_reg:
                from util.contrastive_learning import ContrastiveLearner
                self.netD_cl = ContrastiveLearner(self.netD, image_size=opt.crop_size, hidden_layer=-1,
                                                  input_channels=opt.input_nc)

            # F210929: add Total Variation loss
            self.lambda_tv_loss = self.opt.lambda_tv_loss
            if self.lambda_tv_loss > 0:
                from kornia.losses import total_variation
                self.criterionTV = total_variation

            # F210929: add SSIM loss
            self.lambda_ssim_loss = self.opt.lambda_ssim_loss
            if self.lambda_ssim_loss > 0:
                from kornia.losses import ssim_loss
                self.criterionSSIM = ssim_loss

            # F210929: add PSNR loss
            self.lambda_psnr_loss = self.opt.lambda_psnr_loss
            if self.lambda_psnr_loss > 0:
                from kornia.losses import psnr_loss
                self.criterionPSNR = psnr_loss

    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input (dict): include the data itself and its metadata information.

        The option 'direction' can be used to swap images in domain A and domain B.
        """
        AtoB = self.opt.direction == 'AtoB'
        self.real_A = input['A' if AtoB else 'B'].to(self.device)
        self.real_B = input['B' if AtoB else 'A'].to(self.device)
        self.image_paths = input['A_paths' if AtoB else 'B_paths']

    def set_epoch(self, epoch):
        self.epoch = epoch
        self.iter = 0

    def normalize_input(self):
        """Normalize input data"""
        self.real_AB_min = min(self.real_A.min(), self.real_B.min())
        self.real_AB_max = max(self.real_A.max(), self.real_B.max())

        self.real_A = (self.real_A - self.real_AB_min) / (self.real_AB_max - self.real_AB_min) * 2 - 1
        self.real_B = (self.real_B - self.real_AB_min) / (self.real_AB_max - self.real_AB_min) * 2 - 1

    def denormalize_input(self):
        """De-normalize input data"""
        self.real_A = (self.real_A + 1) / 2 * (self.real_AB_max - self.real_AB_min) + self.real_AB_min
        self.real_B = (self.real_B + 1) / 2 * (self.real_AB_max - self.real_AB_min) + self.real_AB_min

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        self.fake_B = self.netG(self.real_A)  # G(A)

    def backward_D(self):
        """Calculate GAN loss for the discriminator"""
        # Fake; stop backprop to the generator by detaching fake_B
        # we use conditional GANs; we need to feed both input and output to the discriminator
        fake_AB = torch.cat((self.real_A, self.fake_B), 1)
        # pred_fake = self.netD(fake_AB.detach())
        # F210729: add "Differentiable Augmentation for Data-Efficient GAN Training"
        fake_AB = DiffAugment(fake_AB.detach(), policy=self.diff_data_aug)
        pred_fake = self.netD(fake_AB)
        self.loss_D_fake = self.criterionGAN(pred_fake, False)
        # Real
        real_AB = torch.cat((self.real_A, self.real_B), 1)
        # F210729: add "Differentiable Augmentation for Data-Efficient GAN Training"
        real_AB = DiffAugment(real_AB, policy=self.diff_data_aug)
        pred_real = self.netD(real_AB)
        self.loss_D_real = self.criterionGAN(pred_real, True)
        # combine loss and calculate gradients
        self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5
        self.loss_D.backward()

    # F210729: Contrastive Learning regularization
    def backward_D_cl(self):
        """Calculate GAN loss for the discriminator using Contrastive Loss"""
        # Fake; stop backprop to the generator by detaching fake_B
        # we use conditional GANs; we need to feed both input and output to the discriminator

        # update D_cl with generated images
        if self.epoch >= self.cl_reg_gen_epoch:
            # compute output of D_cl
            fake_AB = torch.cat((self.real_A, self.fake_B), 1)
            self.netD_cl(fake_AB.clone().detach(), accumulate=True)

        # Real
        real_AB = torch.cat((self.real_A, self.real_B), 1)
        self.netD_cl(real_AB, accumulate=True)

        # combine loss and calculate gradients
        self.loss_D_cl = self.netD_cl.calculate_loss()
        self.loss_D_cl.backward()

    def backward_G(self):
        """Calculate GAN and L1 loss for the generator"""
        # First, G(A) should fake the discriminator
        fake_AB = torch.cat((self.real_A, self.fake_B), 1)
        # F210729: add "Differentiable Augmentation for Data-Efficient GAN Training"
        fake_AB = DiffAugment(fake_AB, policy=self.diff_data_aug)
        pred_fake = self.netD(fake_AB)

        # F210729: add Top-k GAN Training
        if self.top_k_training:
            k_frac = max(self.top_k_gamma ** (self.epoch + 1), self.top_k_frac)
            k = min(math.ceil(self.batch_size * k_frac), fake_AB.shape[0])
            if k != self.batch_size:
                pred_fake, _ = pred_fake.topk(k=k, largest=False, axis=0)

        self.loss_G_GAN = self.criterionGAN(pred_fake, True)
        # Second, G(A) = B
        self.loss_G_L1 = self.criterionL1(self.fake_B, self.real_B) * self.opt.lambda_L1

        # combine loss and calculate gradients
        self.loss_G = self.loss_G_GAN + self.loss_G_L1

        # F210929: extra losses added
        if self.lambda_msssim_loss or self.lambda_ssim_loss or self.lambda_psnr_loss:
            fake_B_denorm = (self.fake_B + 1) / 2  # * (self.real_AB_max - self.real_AB_min) + self.real_AB_min
            real_B_denorm = (self.real_B + 1) / 2  # * (self.real_AB_max - self.real_AB_min) + self.real_AB_min

        # F210818: add MS-SSIM loss
        if self.lambda_msssim_loss > 0:
            self.loss_G_MSSSIM = self.lambda_msssim_loss * (1 - self.criterionMSSSIM(real_B_denorm, fake_B_denorm)) / 2
            self.loss_G += self.loss_G_MSSSIM

        # F210929: add SSIM loss
        if self.lambda_ssim_loss > 0:
            self.loss_G_SSIM = self.lambda_ssim_loss * self.criterionSSIM(real_B_denorm, fake_B_denorm, window_size=5)
            self.loss_G += self.loss_G_SSIM

        # F210929: add PSNR loss
        if self.lambda_psnr_loss > 0:
            self.loss_G_PSNR = self.lambda_psnr_loss * (-1/self.criterionPSNR(real_B_denorm, fake_B_denorm, max_val=1.))
            self.loss_G += self.loss_G_PSNR

        # F210929: add TV loss
        if self.lambda_tv_loss > 0:
            self.loss_G_TV = self.lambda_tv_loss * self.criterionTV(self.fake_B).mean()
            self.loss_G += self.loss_G_TV

        self.loss_G.backward()

    def optimize_parameters(self):

        # forward pass
        self.forward()  # compute fake images: G(A)

        # F210729: add Contrastive Loss Regularization
        if self.cl_reg:
            self.set_requires_grad(self.netD_cl, True)  # enable backprop for D_cl
            self.optimizer_D.zero_grad()  # set D's gradients to zero
            self.backward_D_cl()  # calculate gradients for D
            self.optimizer_D.step()  # update D's weights

        # update D
        self.set_requires_grad(self.netD, True)  # enable backprop for D
        self.optimizer_D.zero_grad()  # set D's gradients to zero
        self.backward_D()  # calculate gradients for D
        self.optimizer_D.step()  # update D's weights
        # update G
        self.set_requires_grad(self.netD, False)  # D requires no gradients when optimizing G
        self.optimizer_G.zero_grad()  # set G's gradients to zero
        self.backward_G()  # calculate graidents for G
        self.optimizer_G.step()  # udpate G's weights

        self.iter += 1

    def denormalize_visuals(self):
        for visual_name in self.visual_names:
            vis = getattr(self, visual_name)
            vis_unnorm = (vis + 1) / 2 * (self.real_AB_max - self.real_AB_min) + self.real_AB_min
            setattr(self, visual_name, vis_unnorm)
        # return self.get_current_visuals()
