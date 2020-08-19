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
from .base_model import BaseModel
from models.networks import networks
from models.networks.loss import GANLoss, cal_gradient_penalty
from models.networks.utils import get_prior
from util.util import one_hot
from .utils import (
    get_G_heap,
    G_Net
)
from .optimizers import get_optimizer

import copy 
import collections


class EGANModel(BaseModel):

    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """Add new model-specific options and rewrite default values for existing options.

        Parameters:
            parser -- the option parser
            is_train -- if it is training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.
        """
        if is_train:
            parser.add_argument(
                '--g_loss_mode',
                nargs='*',
                default=['nsgan', 'lsgan', 'vanilla'],
                help='lsgan | nsgan | vanilla | wgan | hinge | rsgan',
            )
            parser.add_argument(
                '--d_loss_mode',
                type=str,
                default='lsgan',
                help='lsgan | nsgan | vanilla | wgan | hinge | rsgan',
            )
            parser.add_argument('--which_D', type=str, default='S', help='Standard(S) | Relativistic_average (Ra)') 

            parser.add_argument('--lambda_f', type=float, default=0.1, help='the hyperparameter that balance Fq and Fd')
            parser.add_argument('--candi_num', type=int, default=2, help='# of survived candidatures in each evolutinary iteration.')
            parser.add_argument('--eval_size', type=int, default=64, help='batch size during each evaluation.')
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
        self.output = None
        self.loss_names = ['D_real', 'D_fake', 'D_gp', 'G', 'D']
        self.visual_names = ['real_visual', 'gen_visual']

        if self.isTrain:  # only defined during training time
            self.model_names = ['G', 'D']
        else:
            self.model_names = ['G']

        # define networks 
        self.netG = networks.define_G(opt, self.gpu_ids)
        if self.isTrain:  # only defined during training time
            self.netD = networks.define_D(opt, self.gpu_ids)
            
            # define loss functions
            self.criterionG = None # Will be define by G_mutations
            self.criterionD = None
            # define G mutations 
            self.G_mutations = [
                GANLoss(g_loss, 'G', opt.which_D).to(self.device)
                for g_loss in opt.g_loss_mode
            ]
            # define D adaptions
            self.D_adpations = [
                GANLoss(g_loss, 'D', opt.which_D).to(self.device)
                for g_loss in opt.g_loss_mode
            ]
            # initialize optimizers
            self.optimizer_G = get_optimizer(opt.optim_type)(self.netG.parameters(), lr=opt.lr_g)
            self.optimizer_D = get_optimizer(opt.optim_type)(self.netD.parameters(), lr=opt.lr_d)
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)
        
        # Evolutionary candidatures setting (init)
        self.G_candis = [copy.deepcopy(self.netG.state_dict())] * opt.candi_num
        self.optG_candis = [copy.deepcopy(self.optimizer_G.state_dict())] * opt.candi_num
        self.loss_mode_to_idx = {loss_mode:i for i, loss_mode in enumerate(opt.g_loss_mode)}
        self.wgan_idx = -1
        if "wgan" in self.loss_mode_to_idx:
            self.wgan_idx = self.loss_mode_to_idx["wgan"]


    def forward(self) -> dict:
        batch_size = self.opt.batch_size
        if self.opt.gan_mode == "conditional":
            z = get_prior(self.opt.batch_size, self.opt.z_dim, self.opt.z_type, self.device)
            y = self.CatDis.sample([batch_size])
            y = one_hot(y, [batch_size, self.opt.cat_num])
            gen_data = self.netG(z, y)
            self.set_output(gen_data)
            return {'data': gen_data, 'condition': y}
        elif self.opt.gan_mode == 'unconditional':
            gen_data = self.netG(self.inputs)
            self.set_output(gen_data)
            return {'data': gen_data}
        elif self.opt.gan_mode == 'unconditional-z':
            z = get_prior(self.opt.batch_size, self.opt.z_dim, self.opt.z_type, self.device)
            gen_data = self.netG(z)
            self.set_output(gen_data)
            return {'data': gen_data}
        else:
            raise ValueError(f'unsupported gan_mode {self.opt.gan_mode}')
    
    def set_output(self, x):
        self.output = x

    def get_output(self):
        return self.output

    def backward_G(self, gen_data, criterion) -> dict:
        # pass D
        real_out = self.netD(self.inputs)
        fake_out = self.netD(gen_data)

        loss_G_fake, loss_G_real = criterion(fake_out, real_out) 
        loss_G = loss_G_fake + loss_G_real
        loss_G.backward()

        return {
            '': loss_G,
            'fake': loss_G_fake,
            'real': loss_G_real,
            'mode': self.loss_mode_to_idx[criterion.loss_mode]
        }

    def backward_D(self, gen_data):
        # pass D 
        real_out = self.netD(self.inputs)
        fake_out = self.netD(gen_data)

        self.criterionD = self.D_adpations[self.loss_G["mode"]]
        self.loss_D_fake, self.loss_D_real = self.criterionD(fake_out, real_out)
        if self.loss_G["mode"] == self.wgan_idx and self.opt.use_gp is True:
            self.loss_D_gp = cal_gradient_penalty(
                self.netD,
                self.inputs['data'],
                gen_data['data'],
                self.device,
                type='mixed',
                constant=1.0,
                lambda_gp=10.0,
            )[0]
        else:
            self.loss_D_gp = 0.

        self.loss_D = self.loss_D_fake + self.loss_D_real + self.loss_D_gp
        self.loss_D.backward()

    def optimize_parameters(self):
        if self.step % (self.opt.D_iters + 1) == 0:
            self.set_requires_grad(self.netD, False)
            self.G_candis, self.opt_G_candis, self.loss_G = self.Evo_G(self.G_candis, self.optG_candis)
        else:
            gen_data = self.forward()
            self.set_requires_grad(self.netD, True)
            self.optimizer_D.zero_grad()
            self.backward_D(gen_data)
            self.optimizer_D.step()

        self.step += 1

    def Evo_G(self, G_candis, optG_candis):
        '''
        Enumerate candi_num*G_mutations to find the top 
        candi_num network for fitness_score, self.netG will
        be updated using the best network.
        '''

        G_heap = get_G_heap(self.opt.candi_num)

        # variation-evaluation-selection
        for G_candi, optG_candi in zip(G_candis, optG_candis):
            for criterionG in self.G_mutations: 
                # Variation 
                self.netG.load_state_dict(G_candi)
                self.optimizer_G.load_state_dict(optG_candi)
                self.optimizer_G.zero_grad()
                gen_data = self.forward() 
                G_losses = self.backward_G(gen_data, criterionG)
                self.optimizer_G.step()
                if self.opt.dataset_mode == 'embedding' and not self.opt.exact_orthogonal:
                    self.orthogonalize(self.netG)

                # Evaluation
                with torch.no_grad():
                    eval_data = self.forward()
                fitness = self.fitness_score(eval_data)

                # Selection
                if fitness > G_heap.top().fitness:
                    netG_dict = copy.deepcopy(self.netG.state_dict())
                    optmizerG_dict = copy.deepcopy(self.optimizer_G.state_dict())
                    G_heap.replace(G_Net(fitness=fitness, G_candis=netG_dict, optG_candis=optmizerG_dict, losses=G_losses))
        
        G_candis = [ net.G_candis for net in G_heap.array ]
        optG_candis = [ net.optG_candis for net in G_heap.array ]

        max_idx = G_heap.argmax()

        self.netG.load_state_dict(self.G_candis[max_idx])
        # self.optimizer_G.load_state_dict(self.optG_candis[max_idx]) # not sure if loading is necessary
        loss_G = G_heap.array[max_idx].losses
        return G_candis, optG_candis, loss_G

    def fitness_score(self, eval_data):
        '''
        Evaluate netG based on netD 
        '''
        eval_fake = self.netD(eval_data)

        # Quality fitness score
        Fq = eval_fake.data.mean().item()
        return Fq
