import torch

from .base_model import BaseModel
from models.networks import networks
from models.networks.loss import GANLoss, cal_gradient_penalty
from models.networks.utils import get_prior
from util.util import one_hot
from .optimizers import get_optimizer


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
        self.output = None
        self.loss_names = ['G_real', 'G_fake', 'G_orthogonal', 'D_real', 'D_fake', 'D_gp', 'G', 'D']

        if self.isTrain:  # only defined during training time
            self.model_names = ['G', 'D']
        else:
            self.model_names = ['G']

        # define networks
        self.netG = networks.define_G(opt, self.opt.gpu_ids)
        if self.isTrain:
            self.netD = networks.define_D(opt, self.opt.gpu_ids)

            # define loss functions
            self.criterionG = GANLoss(opt.g_loss_mode, 'G', opt.which_D).to(self.device)
            self.criterionD = GANLoss(opt.d_loss_mode, 'D', opt.which_D).to(self.device)

            # initialize optimizers
            self.optimizer_G = get_optimizer(opt.optim_type)(self.netG.parameters(), lr=opt.lr_g)
            self.optimizer_D = get_optimizer(opt.optim_type)(self.netD.parameters(), lr=opt.lr_d)
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)

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
            gen_data = self.netG({'data': z})
            self.set_output(gen_data)
            return {'data': gen_data}
        else:
            raise ValueError(f'unsupported gan_mode {self.opt.gan_mode}')

    def set_output(self, x):
        self.output = x

    def get_output(self):
        return self.output

    def backward_G(self, gen_data):
        # pass D
        real_out = self.netD(self.inputs)
        fake_out = self.netD(gen_data)
        self.loss_G_fake, self.loss_G_real = self.criterionG(fake_out, real_out)
        if self.opt.dataset_mode == 'embedding' and not self.opt.exact_orthogonal:
            embedding_dim = gen_data['data'].shape[1]
            weight = self.netG.module.layer.data
            self.loss_G_orthogonal = (
                    (weight.T @ weight) - torch.eye(embedding_dim, device=self.device)
            ).norm()
        else:
            self.loss_G_orthogonal = 0.
        self.loss_G = self.loss_G_fake + self.loss_G_real + self.loss_G_orthogonal
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
        if self.step % (self.opt.D_iters + 1) == 0:
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
