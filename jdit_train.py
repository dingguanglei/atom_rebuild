# coding=utf-8
import os, torch
from jdit.Trainer import GanTrainer
from jdit.model import Model
from jdit.optimizer import Optimizer

from mypackage.data import IMAGE_PATH, MASK_PATH_DIC, getDataLoader
from mypackage.tricks import gradPenalty, spgradPenalty, jcbClamp, getPsnr
from mypackage.model.unet import Unet_G
from mypackage.model.wnet import NLayer_D, Wnet_G


class AtomGanTrainer(GanTrainer):
    def __init__(self, nepochs, gpu_ids,
                 netG, netD,
                 optG, optD,
                 train_loader, test_loader=None, cv_loader=None,
                 d_turn=1):
        super(AtomGanTrainer, self).__init__(nepochs, gpu_ids, netG, netD, optG, optD, train_loader,
                                             test_loader=test_loader,
                                             cv_loader=cv_loader,
                                             d_turn=d_turn)

    def compute_d_loss(self):
        d_fake = self.netD(self.fake.detach(), self.input)
        d_real = self.netD(self.ground_truth, self.input)

        var_dic = {}
        var_dic["GP"] = gp = gradPenalty(self.netD, self.ground_truth, self.fake, input=self.input,
                                         use_gpu=self.use_gpu)
        var_dic["SGP"] = sgp = spgradPenalty(self.netD, self.ground_truth, self.fake, input=self.input, type="G",
                                             use_gpu=self.use_gpu) * 0.5 + \
                               spgradPenalty(self.netD, self.ground_truth, self.fake, input=self.input, type="X",
                                             use_gpu=self.use_gpu) * 0.5
        var_dic["WD"] = w_distance = (d_real.mean() - d_fake.mean()).detach()
        var_dic["LOSS_D"] = loss_d = d_fake.mean() - d_real.mean() + gp + sgp

        return loss_d, var_dic

    def compute_g_loss(self):
        d_fake = self.netD(self.fake, self.input)
        var_dic = {}
        var_dic["JC"] = jc = jcbClamp(self.netG, self.input, use_gpu=self.use_gpu)
        var_dic["LOSS_D"] = loss_g = -d_fake.mean() + jc

        return loss_g, var_dic

    def compute_valid(self):
        var_dic = {}
        fake = self.netG(self.input).detach()
        d_fake = self.netD(self.fake, self.input).detach()
        d_real = self.netD(self.ground_truth, self.input).detach()

        var_dic["G"] = loss_g = (-d_fake.mean()).detach()
        var_dic["GP"] = gp = (
            gradPenalty(self.netD, self.ground_truth, self.fake, input=self.input, use_gpu=self.use_gpu)).detach()
        var_dic["D"] = loss_d = (d_fake.mean() - d_real.mean() + gp).detach()
        var_dic["WD"] = w_distance = (d_real.mean() - d_fake.mean()).detach()
        var_dic["PSNR"] = psnr = getPsnr(fake, input, self.use_gpu)
        return var_dic


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "2,3"
    print('===> Check directories')

    gpus = [0]

    d_depth = 32
    g_depth = 32

    batchSize = 2
    test_size = 500
    train_size = None
    cv_size = 500

    nepochs = 130
    d_turn = 1

    lr = 1e-3
    lr_decay = 0.92
    weight_decay = 2e-5
    momentum = 0
    betas = (0.9, 0.999)

    opt_name = "Adam"

    torch.backends.cudnn.benchmark = True
    print('===> Build dataset')
    trainLoader, testLoader, cvLoader = getDataLoader(
        image_dir_path=IMAGE_PATH,
        mask_dir_path=MASK_PATH_DIC["gaussian"],
        batch_size=batchSize,
        test_size=test_size,
        train_size=train_size,
        valid_size=cv_size,
        num_workers=0)

    print('===> Building model')
    net_G = Model(Wnet_G(depth=g_depth, norm_type="switch"),
                  gpu_ids=gpus, use_weights_init=True)

    net_D = Model(NLayer_D(depth=d_depth, norm_type="instance", use_sigmoid=False, use_liner=False),
                  gpu_ids=gpus, use_weights_init=True)

    print('===> Building optimizer')
    optG = Optimizer(net_G.parameters(), lr, lr_decay, weight_decay, momentum, betas, opt_name)
    optD = Optimizer(filter(lambda p: p.requires_grad, net_D.parameters()), lr, lr_decay, weight_decay, momentum, betas,
                     opt_name)
    print('===> Training')
    Trainer = AtomGanTrainer(nepochs, gpus, net_G, net_D, optG, optD, trainLoader, testLoader, cvLoader, d_turn)
    Trainer.train()
