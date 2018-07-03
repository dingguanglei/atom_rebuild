# coding=utf-8
import time
import os
import random
import torch
import copy
from torch.nn import BCELoss, MSELoss, CrossEntropyLoss, BCEWithLogitsLoss
from torch import cuda
from torch.autograd import Variable
from torch.optim import Adam, RMSprop

import torchvision.transforms as transforms
from torchvision.utils import make_grid

from tensorboardX import SummaryWriter

from mypackage.data import IMAGE_PATH, MASK_PATH_DIC, getDataLoader
from mypackage.utils import ElapsedTimer, checkPoint, loadCheckPoint, loadG_Model, writeNetwork
from mypackage.tricks import gradPenalty, spgradPenalty, jcbClamp
from mypackage.model.define import defineNet
from mypackage.model.unet import NLayer_D, Unet_G
from mypackage.model.wnet import NLayer_D, Wnet_G


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
        self.losses = {'G': [], 'D': [], 'GP': [], "SGP": [], 'WD': [], 'GN': [], 'JC': []}
        self.valid_losses = {'G': [], 'D': [], 'WD': [],'JC':[]}
        self.writer = SummaryWriter(log_dir="log")
        self.lr = 2e-3
        self.lr_decay = 0.94
        self.weight_decay = 2e-5
        self.opt_g = Adam(self.netG.parameters(), lr=self.lr, betas=(0.9, 0.99), weight_decay=self.weight_decay)
        self.opt_d = RMSprop(self.netD.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        input = torch.autograd.Variable(torch.Tensor(4, 1, 256, 256), requires_grad=True)
        net_cpu = copy.deepcopy(self.netG.module)
        self.writer.add_graph(model=net_cpu.cpu(), input_to_model=input)

    def _dis_train_iteration(self, input, fake, real):
        self.opt_d.zero_grad()

        d_fake = self.netD(fake, input)
        d_real = self.netD(real, input)

        gp = gradPenalty(self.netD, real, fake, input=input)
        sgp = spgradPenalty(self.netD, input, real, fake, type="G") * 0.5 + \
              spgradPenalty(self.netD, input, real, fake, type="X") * 0.5
        loss_d = d_fake.mean() - d_real.mean() + gp + sgp
        w_distance = (d_real.mean() - d_fake.mean()).detach()
        loss_d.backward()
        self.opt_d.step()

        self.losses["D"].append(loss_d.detach())
        self.losses["WD"].append(w_distance.detach())
        self.losses["GP"].append(gp.detach())
        self.losses["SGP"].append(sgp.detach())
        self._watchLoss(["D", "GP", "WD", "SGP"], loss_dic=self.losses, type="Train")
        d_log = "Loss_D: {:.4f}".format(loss_d.detach())
        return d_log

    def _gen_train_iteration(self, input):
        self.opt_g.zero_grad()

        fake = self.netG(input)
        d_fake = self.netD(fake, input)
        jc = jcbClamp(self.netG, input, use_gpu=self.use_gpus)
        loss_g = -d_fake.mean() + jc
        loss_g.backward()
        self.opt_g.step()
        self.losses["JC"].append(jc.detach())
        self.losses["G"].append(loss_g.detach())
        self._watchLoss(["JC"], loss_dic=self.losses, type="Train")
        self._watchLoss(["G"], loss_dic=self.losses, type="Train")
        g_log = "Loss_G: {:.4f}".format(loss_g.detach())
        return g_log

    def _train_epoch(self, input, real):
        epoch = self.epoch
        for iteration, batch in enumerate(self.train_loader, 1):
            timer = ElapsedTimer()
            self.count += 1

            real_a_cpu, real_b_cpu = batch[0], batch[1]
            input.data.resize_(real_a_cpu.size()).copy_(real_a_cpu)  # input data
            real.data.resize_(real_b_cpu.size()).copy_(real_b_cpu)  # real data
            fake = self.netG(input)

            d_log = self._dis_train_iteration(input, fake.detach(), real)
            g_log = self._gen_train_iteration(input)

            print("===> Epoch[{}]({}/{}): {}\t{} ".format(
                epoch, iteration, self.steps, d_log, g_log))
            one_step_cost = time.time() - timer.start_time
            left_time_one_epoch = timer.elapsed((self.steps - iteration) * one_step_cost)
            print("leftTime: %s" % left_time_one_epoch)

            if iteration == 1:
                self._watchImg(input, fake, real, type="Train")

    def train(self):
        input = Variable()
        real = Variable()
        if self.use_gpus:
            input = input.cuda()
            real = real.cuda()
        startEpoch = 1
        # netG, netD = loadCheckPoint(netG, netD, startEpoch)
        for epoch in range(startEpoch, nepochs + 1):
            self.epoch = epoch
            timer = ElapsedTimer()
            self._train_epoch(input, real)
            self.valid()
            self._watchNetParams(self.netG, epoch)
            left_time = timer.elapsed((nepochs - epoch) * (time.time() - timer.start_time))
            print("leftTime: %s" % left_time)
            if epoch % 2 == 0:
                self.lr = self.lr * self.lr_decay
                self.opt_g = Adam(net_G.parameters(), lr=self.lr, betas=(0.9, 0.99), weight_decay=weight_decay)
                self.opt_d = RMSprop(net_D.parameters(), lr=self.lr, weight_decay=weight_decay)
                print("change learning rate to %s" % self.lr)
            if epoch % 10 == 0:
                self.predict()
                checkPoint(net_G, net_D, epoch)
        self.writer.close()

    def _watchNetParams(self, net, count):
        for name, param in net.named_parameters():
            if "bias" in name:
                continue
            self.writer.add_histogram(name, param.clone().cpu().data.numpy(), count)

    def _watchLoss(self, loss_keys, loss_dic, type="Train"):
        for key in loss_keys:
            self.writer.add_scalars(key, {type: loss_dic[key][-1]}, self.count)

    def _watchImg(self, input, fake, real, type="Train", show_imgs_num=3):
        out = None
        input_torch = None
        prediction_torch = None
        real_torch = None
        batchSize = input.shape[0]
        show_nums = min(show_imgs_num,batchSize)
        randindex_list = random.sample(list(range(batchSize)), show_nums)
        for randindex in randindex_list:
            input_torch = input[randindex].cpu().detach()
            input_torch = transforms.Normalize([-1], [2])(input_torch)

            prediction_torch = fake[randindex].cpu().detach()
            prediction_torch = transforms.Normalize([-1], [2])(prediction_torch)

            real_torch = real[randindex].cpu().detach()
            real_torch = transforms.Normalize([-1], [2])(real_torch)
            out_1 = torch.stack((input_torch, prediction_torch, real_torch))
            if out is None:
                out = out_1
            else:
                out = torch.cat((out_1, out))
        out = make_grid(out, nrow=3)
        self.writer.add_image('%s-in-pred-real' % type, out, self.epoch)

        input = transforms.ToPILImage()(input_torch).convert("L")
        prediction = transforms.ToPILImage()(prediction_torch).convert("L")
        real = transforms.ToPILImage()(real_torch).convert("L")

        in_filename = "plots/%s/E%03d_in_.png" % (type, self.epoch)
        real_filename = "plots/%s/E%03d_real_.png" % (type, self.epoch)
        out_filename = "plots/%s/E%03d_out_.png" % (type, self.epoch)
        input.save(in_filename)
        prediction.save(out_filename)
        real.save(real_filename)

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
            self._watchImg(input, fake, real, type="Test", show_imgs_num=8)

        self.netG.train()

    def valid(self):
        avg_loss_g = 0
        avg_loss_d = 0
        avg_w_distance = 0
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
            gp = gradPenalty(self.netD, real, fake, input=input)
            loss_d = d_fake.mean() - d_real.mean() + gp
            w_distance = d_real.mean() - d_fake.mean()
            # 求和
            avg_w_distance += w_distance.detach()
            avg_loss_d += loss_d.detach()
            avg_loss_g += loss_g.detach()
        avg_w_distance = avg_w_distance / len_test_data
        avg_loss_d = avg_loss_d / len_test_data
        avg_loss_g = avg_loss_g / len_test_data
        self.valid_losses["WD"].append(avg_w_distance)
        self.valid_losses["D"].append(avg_loss_d)
        self.valid_losses["G"].append(avg_loss_g)
        # print("===> CV_Loss_D: {:.4f} CV_WD:{:.4f} CV_Loss_G: {:.4f}".format(avg_loss_d, avg_w_distance, avg_loss_g))
        self._watchLoss(["D", "G", "WD"], loss_dic=self.valid_losses, type="Valid")
        self._watchImg(input, fake, real, type="Valid")
        self.netG.train()
        self.netD.train()
        self.netG.zero_grad()
        self.netD.zero_grad()

        return avg_w_distance


def buildDir():
    dirs = ["plots", "plots/Test", "plots/Train", "plots/Valid", "checkpoint"]
    for dir in dirs:
        if not os.path.exists(dir):
            print("%s directory is not found. Build now!" % dir)
            os.mkdir(dir)


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
    print('===> Check directories')
    buildDir()

    gpus = (0, 1)
    # gpus = []
    d_depth = 32
    g_depth = 32
    lr = 2e-3
    beta1 = 0.9
    weight_decay = 2e-5

    batchSize = 8
    nepochs = 100

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
    # net_G = defineNet(Unet_G(depth=g_depth, norm_type="instance"),
    #                   gpu_ids=gpus, use_weights_init=True)
    net_G = defineNet(Wnet_G(depth=g_depth, active_type="LeakyReLU", norm_type="instance"),
                      gpu_ids=gpus, use_weights_init=True)
    net_D = defineNet(NLayer_D(depth=d_depth, norm_type="instance", use_sigmoid=False, use_liner=False),
                      gpu_ids=gpus, use_weights_init=True)

    print('===> Training')
    Trainer = Trainer(net_G, net_D, trainLoader, testLoader, cvLoader, gpus)
    Trainer.train()
