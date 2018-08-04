# coding=utf-8
import time
import os
import torch
import copy
from torch.autograd import Variable
from torch.optim import Adam, RMSprop

from mypackage.data import IMAGE_PATH, MASK_PATH_DIC, getDataLoader
from mypackage.utils import ElapsedTimer, checkPoint, Watcher, buildDir
from mypackage.tricks import gradPenalty, spgradPenalty, jcbClamp, getPsnr
from mypackage.model.define import defineNet
from mypackage.model.unet import Unet_G
from mypackage.model.wnet import NLayer_D, Wnet_G,Attn_Discriminator,Attn_Wnet_G
from mypackage.optimizer import Optimizer


class Trainer(object):
    def __init__(self, netG, netD, train_loader, test_loader=None, cv_loader=None, gpus=()):
        self.use_gpus = torch.cuda.is_available() & (len(gpus) > 0)
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.cv_loader = cv_loader
        self.netG = netG
        self.netD = netD
        self.count = 0
        self.steps = len(train_loader)
        self.watcher = Watcher(logdir="log")

        self.lr = 1e-3
        self.lr_decay = 0.8
        self.weight_decay = 2e-5
        self.betas = (0.9, 0.99)
        self.opt_g = Optimizer(filter(lambda p: p.requires_grad, self.netG.parameters()), self.lr, self.lr_decay, self.weight_decay, betas=self.betas,
                               opt_name="Adam")
        # self.opt_g = Adam(self.netG.parameters(), lr=self.lr, betas=(0.9, 0.999), weight_decay=self.weight_decay)
        self.opt_d = Optimizer(filter(lambda p: p.requires_grad, self.netD.parameters()), self.lr, self.lr_decay,
                               self.weight_decay, opt_name="RMSprop")
        # self.opt_ds = RMSprop(filter(lambda p: p.requires_grad, self.netD.parameters()), lr=self.lr,
        #                      weight_decay=self.weight_decay)
        self.nepochs = 150 * 5

        net_cpu = copy.deepcopy(self.netG).cpu()
        self.watcher.graph(net_cpu, input_shape=(2, 1, 256, 256))

    def _dis_train_iteration(self, input, fake, real):
        self.opt_d.zero_grad()

        d_fake = self.netD(fake, input)
        d_real = self.netD(real, input)

        gp = gradPenalty(self.netD, real, fake, input=input, use_gpu=self.use_gpus)
        sgp = spgradPenalty(self.netD, input, real, fake, type="G", use_gpu=self.use_gpus) * 0.5\
              + spgradPenalty(self.netD, input, real, fake, type="X", use_gpu=self.use_gpus) * 0.5
        loss_d = d_fake.mean() - d_real.mean() + gp + sgp
        w_distance = (d_real.mean() - d_fake.mean()).detach()
        loss_d.backward()
        self.opt_d.step()
        self.watcher.scalars(["D", "GP", "WD", "SGP"], [loss_d, gp, w_distance, sgp], self.count, tag="Train")
        d_log = "Loss_D: {:.4f}".format(loss_d.detach())
        return d_log

    def _gen_train_iteration(self, input, fake=None):
        self.opt_g.zero_grad()
        if fake is None:
            fake = self.netG(input)
        d_fake = self.netD(fake, input)
        # jc = jcbClamp(self.netG, input, use_gpu=self.use_gpus)
        loss_g = -d_fake.mean()
        loss_g.backward()
        self.opt_g.step()
        psnr = getPsnr(fake, input, self.use_gpus)
        self.watcher.scalars(["PSNR", "G"], [psnr, loss_g], self.count, tag="Train")
        g_log = "Loss_G: {:.4f} PSNR:{:.4f}".format(loss_g.detach(), psnr)
        return g_log

    def _train_epoch(self, input, real):
        epoch = self.epoch
        d_turn = 5
        for iteration, batch in enumerate(self.train_loader, 1):
            timer = ElapsedTimer()
            self.count += 1

            real_a_cpu, real_b_cpu = batch[0], batch[1]
            input.data.resize_(real_a_cpu.size()).copy_(real_a_cpu)  # input data
            real.data.resize_(real_b_cpu.size()).copy_(real_b_cpu)  # real data
            fake = self.netG(input)

            d_log = self._dis_train_iteration(input, fake.detach(), real)
            g_log = ""
            if (self.count <= 75 and self.count  % 25 == 0) or (self.count > 75 and self.count % d_turn == 0):
                g_log = self._gen_train_iteration(input, fake)

            # d_log = self._dis_train_iteration(input, fake.detach(), real)
            # g_log = self._gen_train_iteration(input, fake)

            print("===> Epoch[{}]({}/{}): {}\t{} ".format(
                epoch, iteration, self.steps, d_log, g_log))

            one_step_cost = time.time() - timer.start_time
            left_time_one_epoch = timer.elapsed((self.steps - iteration) * one_step_cost)
            print("leftTime: %s" % left_time_one_epoch)
            if iteration == 1:
                self.watcher.images([input, fake, real], ["input", "fake", "real"], self.epoch, tag="Train",
                                    show_imgs_num=3,
                                    mode="L",
                                    mean= -1,
                                    std = 2)

    def train(self):
        input = Variable()
        real = Variable()
        if self.use_gpus:
            input = input.cuda()
            real = real.cuda()
        startEpoch = 1
        # netG, netD = loadCheckPoint(netG, netD, startEpoch)
        for epoch in range(startEpoch, self.nepochs + 1):
            self.epoch = epoch
            timer = ElapsedTimer()
            self._train_epoch(input, real)
            self.valid()
            # self.watcher.netParams(self.netG, epoch)
            left_time = timer.elapsed((self.nepochs - epoch) * (time.time() - timer.start_time))
            print("leftTime: %s" % left_time)
            # if epoch == 1:
            #     self.opt_g.do_lr_decay(self.netG.parameters(), reset_lr=1e-4)
            #     self.opt_d.do_lr_decay(filter(lambda p: p.requires_grad, self.netD.parameters()), reset_lr=1e-3)
            # elif epoch == 40:
            #     self.opt_g.do_lr_decay(self.netG.parameters(), reset_lr=1e-5)
            #     self.opt_d.do_lr_decay(filter(lambda p: p.requires_grad, self.netD.parameters()), reset_lr=1e-4)
            # elif epoch == 80:
            #     self.opt_g.do_lr_decay(self.netG.parameters(), reset_lr=1e-6)
            #     self.opt_d.do_lr_decay(filter(lambda p: p.requires_grad, self.netD.parameters()), reset_lr=1e-5)
            if epoch % 10 == 0:
                self.lr = self.lr * self.lr_decay
                self.opt_g.do_lr_decay(filter(lambda p: p.requires_grad, self.netG.parameters()))
                self.opt_d.do_lr_decay(filter(lambda p: p.requires_grad, self.netD.parameters()))
                # self.opt_g = Adam(net_G.parameters(), lr=self.lr, betas=(0.9, 0.99), weight_decay=self.weight_decay)
                # self.opt_d = RMSprop(filter(lambda p: p.requires_grad, self.netD.parameters()), lr=self.lr,
                #                      weight_decay=self.weight_decay)
                print("change learning rate to %s" % self.lr)
            if epoch % 10 == 0:
                self.predict()
                checkPoint(net_G, net_D, epoch, name="")
        self.watcher.close()

    def predict(self):
        for input, real in self.test_loader:
            input = Variable(input)
            real = Variable(real)
            if self.use_gpus:
                input = input.cuda()
                real = real.cuda()
            self.netG.eval()
            fake = self.netG(input).detach().detach()
            self.netG.zero_grad()
            self.watcher.images([input, fake, real], ["input", "fake", "real"], self.epoch, tag="Test",
                                show_imgs_num=8,
                                mode="L",
                                mean= -1,
                                std = 2)
        self.netG.train()

    def valid(self):
        avg_loss_g = 0
        avg_loss_d = 0
        avg_w_distance = 0
        avg_psnr = 0
        # netG = netG._d
        self.netG.eval()
        self.netD.eval()
        input = Variable()
        real = Variable()
        if self.use_gpus:
            input = input.cuda()
            real = real.cuda()
        len_test_data = len(self.cv_loader)
        for iteration, batch in enumerate(self.cv_loader, 1):
            input.data.resize_(batch[0].size()).copy_(batch[0])  # input data
            real.data.resize_(batch[1].size()).copy_(batch[1])  # real data
            ## 计算G的LOSS
            fake = self.netG(input).detach()
            d_fake = self.netD(fake, input).detach()
            loss_g = -d_fake.mean()

            # 计算D的LOSS
            d_real = self.netD(real, input).detach()
            gp = gradPenalty(self.netD, real, fake, input=input, use_gpu=self.use_gpus)
            loss_d = d_fake.mean() - d_real.mean() + gp
            w_distance = d_real.mean() - d_fake.mean()
            psnr = getPsnr(fake, input, self.use_gpus)

            # 求和
            avg_w_distance += w_distance.detach()
            avg_psnr += psnr
            avg_loss_d += loss_d.detach()
            avg_loss_g += loss_g.detach()
        avg_w_distance = avg_w_distance / len_test_data
        avg_loss_d = avg_loss_d / len_test_data
        avg_loss_g = avg_loss_g / len_test_data
        avg_psnr = avg_psnr / len_test_data
        self.watcher.scalars(["D", "G", "WD", "PSNR"], [avg_loss_d, avg_loss_g, avg_w_distance, avg_psnr], self.count,
                             tag="Valid")
        self.watcher.images([input, fake, real], ["input", "fake", "real"], self.epoch, tag="Valid",mean= -1,
                                    std = 2)
        self.netG.train()
        self.netD.train()
        self.netG.zero_grad()
        self.netD.zero_grad()


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "2,3"
    print('===> Check directories')
    buildDir()

    gpus = (0, 1)
    d_depth = 32
    g_depth = 32
    batchSize = 32
    test_size = 500
    train_size = None
    cv_size = 500

    torch.backends.cudnn.benchmark = True

    print('===> Build dataset')
    trainLoader, testLoader, cvLoader = getDataLoader(
        image_dir_path=IMAGE_PATH,
        mask_dir_path=MASK_PATH_DIC["gaussian"],
        batch_size=batchSize,
        test_size=test_size,
        train_size=train_size,
        valid_size=cv_size)

    print('===> Building model')
    # net_G = defineNet(Wnet_G(depth=g_depth, norm_type="instance",active_type="LeakyReLU"),
    #                   gpu_ids=gpus, use_weights_init=True)
    # net_G = defineNet(Wnet_G(depth=g_depth, active_type="LeakyReLU", norm_type="switch"),
    #                   gpu_ids=gpus, use_weights_init=True)
    # net_D = defineNet(NLayer_D(depth=d_depth, norm_type="instance", use_sigmoid=False, use_liner=False),
    #                   gpu_ids=gpus, use_weights_init=True)
    net_G = defineNet(Attn_Wnet_G(depth=g_depth, norm_type="batch", active_type="ReLU"),
                                        gpu_ids=gpus, use_weights_init=True)

    net_D = defineNet(Attn_Discriminator(depth=d_depth),
                                        gpu_ids=gpus, use_weights_init=True)

    print('===> Training')
    Trainer = Trainer(net_G, net_D, trainLoader, testLoader, cvLoader, gpus)
    Trainer.train()
