import torch
from .base_model import BaseModel
from networks import networks
from networks.loss import GANLoss, cal_gradient_penalty
from networks.utils import get_prior
from util.util import one_hot


class TwoPlayerGANModel(BaseModel):

    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        if is_train:
            parser.add_argument('--g_loss_mode', type=str, default='lsgan',
                                help='lsgan | nsgan | vanilla | wgan | hinge | rsgan')
            parser.add_argument('--d_loss_mode', type=str, default='lsgan',
                                help='lsgan | nsgan | vanilla | wgan | hinge | rsgan')
            parser.add_argument('--which_D', type=str, default='S', help='Standard(S) | Relativistic_average (Ra)')

        return parser

    def __init__(self, opt):
        BaseModel.__init__(self, opt)

        self.loss_names = ['G_real', 'G_fake', 'D_real', 'D_fake', 'D_gp', 'G', 'D']
        self.visual_names = ['real_visual', 'gen_visual']

        if self.isTrain:  # only defined during training time
            self.model_names = ['G', 'D']
        else:
            self.model_names = ['G']

        # define networks
        self.netG = networks.define_G(opt, self.device)
        if self.isTrain:
            self.netD = networks.define_D(opt, self.device)

            # define loss functions
            self.criterionG = GANLoss(opt.g_loss_mode, 'G', opt.which_D).to(self.device)
            self.criterionD = GANLoss(opt.d_loss_mode, 'D', opt.which_D).to(self.device)

            # initialize optimizers
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=opt.lr_g, betas=(opt.beta1, opt.beta2))
            self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=opt.lr_d, betas=(opt.beta1, opt.beta2))
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)

    def forward(self) -> dict:
        batch_size = self.opt.batch_size
        if self.opt.gan_mode == "conditional":
            z = get_prior(self.opt.batch_size, self.opt.z_dim, self.opt.z_type, self.device)
            y = self.CatDis.sample([batch_size])
            y = one_hot(y, [batch_size, self.opt.cat_num])
            gen_data = self.netG(z, y)
            return {'data': gen_data, 'condition': y}
        elif self.opt.gan_mode == 'unconditional':
            source = self.inputs['source']
            gen_data = self.netG(source)
            return {'data': gen_data}
        elif self.opt.gan_mode == 'unconditional-z':
            z = get_prior(self.opt.batch_size, self.opt.z_dim, self.opt.z_type, self.device)
            gen_data = self.netG(z)
            return {'data': gen_data}
        else:
            raise ValueError(f'unsupported gan_mode {self.opt.gan_mode}')

    def backward_G(self, gen_data):
        # pass D
        real_out = self.netD(self.inputs)
        fake_out = self.netD(gen_data)

        self.loss_G_fake, self.loss_G_real = self.criterionG(fake_out, real_out)
        self.loss_G = self.loss_G_fake + self.loss_G_real
        self.loss_G.backward()

    def backward_D(self, gen_data):
        # pass D
        real_out = self.netD(self.inputs)
        fake_out = self.netD(gen_data)

        self.loss_D_fake, self.loss_D_real = self.criterionD(fake_out, real_out)
        if self.opt.use_gp is True:
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
        gen_data = self.forward()
        if self.step == 0:
            self.set_requires_grad(self.netD, False)
            self.optimizer_G.zero_grad()
            self.backward_G(gen_data)
            self.optimizer_G.step()
        else:
            self.set_requires_grad(self.netD, True)
            self.optimizer_D.zero_grad()
            self.backward_D(gen_data)
            self.optimizer_D.step()

        self.step += 1
