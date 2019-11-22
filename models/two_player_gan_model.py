"""Model class template

This module provides a template for users to implement custom models.
You can specify '--model template' to use this model.
The class name should be consistent with both the filename and its model option.
The filename should be <model>_dataset.py
The class name should be <Model>Dataset.py
It implements a simple image-to-image translation baseline based on regression loss.
Given input-output pairs (data_A, data_B), it learns a network netG that can minimize the following L1 loss:
    min_<netG> ||netG(data_A) - data_B||_1
You need to implement the following functions:
    <modify_commandline_options>:　Add model-specific options and rewrite default values for existing options.
    <__init__>: Initialize this model class.
    <set_input>: Unpack input data and perform data pre-processing.
    <forward>: Run forward pass. This will be called by both <optimize_parameters> and <test>.
    <optimize_parameters>: Update network weights; it will be called in every training iteration.
"""
import torch
import numpy as np
from .base_model import BaseModel
from networks import networks
from networks.loss import GANLoss, cal_gradient_penalty
from util.util import one_hot
from torch.distributions import Categorical


class TwoPlayerGANModel(BaseModel):
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """Add new model-specific options and rewrite default values for existing options.

        Parameters:
            parser -- the option parser
            is_train -- if it is training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.
        """
        # parser.set_defaults(dataset_mode='aligned')  # You can rewrite default values for this model. For example, this model usually uses aligned dataset as its dataset
        if is_train:
            parser.add_argument('--g_loss_mode', type=str, default='lsgan',
                                help='lsgan | nsgan | vanilla | wgan | hinge | rsgan')
            parser.add_argument('--d_loss_mode', type=str, default='lsgan',
                                help='lsgan | nsgan | vanilla | wgan | hinge | rsgan')
            parser.add_argument('--which_D', type=str, default='S', help='Standard(S) | Relativistic_average (Ra)')

        return parser

    def __init__(self, opt):
        """Initialize this model class.

        Parameters:
            opt -- training/test options

        A few things can be done here.
        - (required) call the initialization function of BaseModel
        - define loss function, visualization images, model names, and optimizers
        """
        BaseModel.__init__(self, opt)  # call the initialization method of BaseModel

        self.opt = opt
        if opt.d_loss_mode == 'wgan' and not opt.use_gp:
            raise NotImplementedError('using wgan on D must be with use_gp = True.')

        self.loss_names = ['G_real', 'G_fake', 'D_real', 'D_fake', 'D_gp', 'G', 'D']
        self.visual_names = ['real_visual', 'gen_visual']

        if self.isTrain:  # only defined during training time
            self.model_names = ['G', 'D']
        else:
            self.model_names = ['G']

        if self.opt.cgan:
            probs = np.ones(self.opt.cat_num) / self.opt.cat_num
            self.CatDis = Categorical(torch.tensor(probs))
        # define networks 
        self.netG = networks.define_G(opt, self.device)
        if self.isTrain:
            # define a discriminator;
            # conditional GANs need to take both input and output images;
            # Therefore, #channels for D is input_nc + output_nc
            self.netD = networks.define_D(opt, self.device)
        if self.isTrain:  # only defined during training time
            # define loss functions
            self.criterionG = GANLoss(opt.g_loss_mode, 'G', opt.which_D).to(self.device)
            self.criterionD = GANLoss(opt.d_loss_mode, 'D', opt.which_D).to(self.device)
            # initialize optimizers
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=opt.lr_g, betas=(opt.beta1, opt.beta2))
            self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=opt.lr_d, betas=(opt.beta1, opt.beta2))
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)
        # visualize settings
        self.N = int(np.trunc(np.sqrt(min(opt.batch_size, 64))))
        if self.opt.z_type == 'Gaussian':
            self.z_fixed = torch.randn(self.N * self.N, opt.z_dim, 1, 1, device=self.device)
        elif self.opt.z_type == 'Uniform':
            self.z_fixed = torch.rand(self.N * self.N, opt.z_dim, 1, 1, device=self.device) * 2. - 1.
        if self.opt.cgan:
            yf = self.CatDis.sample([self.N * self.N])
            self.y_fixed = one_hot(yf, [self.N * self.N, self.opt.cat_num])

    def forward(self, batch_size=None):
        bs = self.opt.batch_size if batch_size is None else batch_size
        if self.opt.z_type == 'Gaussian':
            z = torch.randn(bs, self.opt.z_dim, 1, 1, device=self.device)
        elif self.opt.z_type == 'Uniform':
            z = torch.rand(bs, self.opt.z_dim, 1, 1, device=self.device) * 2. - 1.

        if not self.opt.cgan:
            self.gen_imgs = self.netG(z)
        else:
            y = self.CatDis.sample([bs])
            self.y_ = one_hot(y, [bs, self.opt.cat_num])
            self.gen_imgs = self.netG(z, self.y_)

    def backward_G(self):
        # pass D 
        if not self.opt.cgan:
            self.fake_out = self.netD(self.gen_imgs)
            self.real_out = self.netD(self.real_imgs)
        else:
            self.fake_out = self.netD(self.gen_imgs, self.y_)
            self.real_out = self.netD(self.real_imgs, self.targets)

        self.loss_G_fake, self.loss_G_real = self.criterionG(self.fake_out, self.real_out)
        self.loss_G = self.loss_G_fake + self.loss_G_real
        self.loss_G.backward()

    def backward_D(self):
        self.gen_imgs = self.gen_imgs.detach()
        # pass D 
        if not self.opt.cgan:
            self.fake_out = self.netD(self.gen_imgs)
            self.real_out = self.netD(self.real_imgs)
        else:
            self.fake_out = self.netD(self.gen_imgs, self.y_)
            self.real_out = self.netD(self.real_imgs, self.targets)

        self.loss_D_fake, self.loss_D_real = self.criterionD(self.fake_out, self.real_out)
        if self.opt.use_gp is True:
            self.loss_D_gp = \
            cal_gradient_penalty(self.netD, self.real_imgs, self.gen_imgs, self.device, type='mixed',
                                          constant=1.0, lambda_gp=10.0)[0]
        else:
            self.loss_D_gp = 0.

        self.loss_D = self.loss_D_fake + self.loss_D_real + self.loss_D_gp
        self.loss_D.backward()

    def optimize_parameters(self):
        for i in range(self.opt.D_iters + 1):
            self.real_imgs = self.input_imgs[i * self.opt.batch_size:(i + 1) * self.opt.batch_size, :, :, :]
            if self.opt.cgan:
                self.targets = self.input_target[i * self.opt.batch_size:(i + 1) * self.opt.batch_size, :]
            self.forward()
            # update G
            if i == 0:
                self.set_requires_grad(self.netD, False)
                self.optimizer_G.zero_grad()
                self.backward_G()
                self.optimizer_G.step()
            # update D
            else:
                self.set_requires_grad(self.netD, True)
                self.optimizer_D.zero_grad()
                self.backward_D()
                self.optimizer_D.step()
