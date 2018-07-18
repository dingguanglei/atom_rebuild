# coding=utf-8
import random
from torch import save, load
from torch.nn import DataParallel
import torch
import time, os
from tensorboardX import SummaryWriter
import torchvision.transforms as transforms
from torchvision.utils import make_grid


class ElapsedTimer(object):
    def __init__(self):
        self.start_time = time.time()

    def elapsed(self, sec):
        if sec < 60:
            return str(round(sec, 2)) + " sec"
        elif sec < (60 * 60):
            return str(round(sec / 60, 2)) + " min"
        else:
            return str(round(sec / (60 * 60), 2)) + " hr"

    def elapsed_time(self):
        print("Elapsed: %s " % self.elapsed(time.time() - self.start_time))


def loadModel(model_path, model_weights_path, gpus=None):
    print("load model uses CPU...")
    model = torch.load(model_path, map_location=lambda storage, loc: storage)
    print("load weights uses CPU...")
    weights = torch.load(model_weights_path, map_location=lambda storage, loc: storage)

    if model.module:
        print("deal with dataparallel and extract module...")
        model = model.module
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in weights.items():
            name = k[7:]  # remove `module.`
            new_state_dict[name] = v
        weights = new_state_dict
        # load params

    model.load_state_dict(weights)
    if torch.cuda.is_available() and (len(gpus) == 1):
        print("convert to GPU %s" % str(gpus))
        model = model.cuda()
    elif torch.cuda.is_available() and (len(gpus) > 1):
        print("convert to GPUs %s" % str(gpus))
        model = DataParallel(model, gpus).cuda()

    return model.eval()


class Model():
    def __init__(self, epoch=30, gpus=None, model_path=None, model_weights_path=None, name=""):
        self.epoch = epoch
        self.model_path = "checkpoint/{}Model_G_{}.pth".format(name, self.epoch)
        self.model_weights_path = "checkpoint/{}Model_weights_G_{}.pth".format(name, self.epoch)
        self.model = self.loadModel(gpus)

    def loadModel(self, gpus=None):
        model_path = self.model_path
        model_weights_path = self.model_weights_path
        print("load model uses CPU...")
        model = torch.load(model_path, map_location=lambda storage, loc: storage)
        print("load weights uses CPU...")
        weights = torch.load(model_weights_path, map_location=lambda storage, loc: storage)

        if model.module:
            print("deal with dataparallel...")
            model = model.module
            from collections import OrderedDict
            new_state_dict = OrderedDict()
            for k, v in weights.items():
                name = k[7:]  # remove `module.`
                new_state_dict[name] = v
            weights = new_state_dict
            # load params

        model.load_state_dict(weights)
        if torch.cuda.is_available() and (len(gpus) == 1):
            print("convert to GPU %s" % str(gpus))
            model = model.cuda()
        elif torch.cuda.is_available() and (len(gpus) > 1):
            print("convert to GPUs %s" % str(gpus))
            model = DataParallel(model, gpus).cuda()
        # if torch.cuda.is_available() and (gpus is not None):
        #     print("load model uses GPU")
        #     model = torch.load(model_path, map_location=lambda storage, loc: storage.cuda(gpus))
        #     # model = torch.load(model_path, map_location=lambda storage, loc: storage)
        #     if model.module:
        #         model = model.module
        #     weights = torch.load(model_weights_path, map_location=lambda storage, loc: storage.cuda(gpus))
        #     model.load_state_dict(weights)
        # else:
        #     print("load model uses CPU")
        #     model = torch.load(model_path, map_location=lambda storage, loc: storage)
        #     if model.module:
        #         model = model.module
        #     weights = torch.load(model_weights_path, map_location=lambda storage, loc: storage)
        #
        #     from collections import OrderedDict
        #     new_state_dict = OrderedDict()
        #     for k, v in weights.items():
        #         name = k[7:]  # remove `module.`
        #         new_state_dict[name] = v
        #     # load params
        #     model.load_state_dict(new_state_dict)

        return model.eval()


def checkPoint(netG, netD, epoch, name=""):
    if not os.path.exists("checkpoint"):
        os.mkdir("checkpoint")
    g_model_weights_path = "checkpoint/{}Model_weights_G_{}.pth".format(name, epoch)
    d_model_weights_path = "checkpoint/{}Model_weights_D_{}.pth".format(name, epoch)
    g_model_path = "checkpoint/{}Model_G_{}.pth".format(name, epoch)
    d_model_path = "checkpoint/{}Model_D_{}.pth".format(name, epoch)
    save(netG.state_dict(), g_model_weights_path)
    save(netD.state_dict(), d_model_weights_path)
    save(netG, g_model_path)
    save(netD, d_model_path)
    print("Checkpoint saved !")


# def loadCheckPoint(netG, netD, epoch, name=""):
#     net_g_model_out_path = "checkpoint/{}Model_weights_G_{}.pth".format(name, epoch)
#     net_d_model_out_path = "checkpoint/{}Model_weights_D_{}.pth".format(name, epoch)
#     netG.load_state_dict(load(net_g_model_out_path))
#     netD.load_state_dict(load(net_d_model_out_path))
#     print("Checkpoint loaded !")
#     return netG, netD
#
#
# def loadG_Model(epoch):
#     model_out_path = "checkpoint/Model_G_{}.pth".format(epoch)
#     G_model = load(model_out_path)
#     return G_model

class Watcher(object):
    def __init__(self, logdir="log"):
        self.writer = SummaryWriter(log_dir=logdir)

    def watchNetParams(self, network, global_step):
        for name, param in network.named_parameters():
            if "bias" in name:
                continue
            self.writer.add_histogram(name, param.clone().cpu().data.numpy(), global_step, bins="auto")

    def watchLoss(self, loss_keys, loss_dic, global_step, tag="Train"):
        for key in loss_keys:
            self.writer.add_scalars(key, {tag: loss_dic[key][-1]}, global_step)

    def watchImg(self, input, fake, real, global_step, tag="Train", show_imgs_num=3, mode="L"):
        out = None
        input_torch = None
        prediction_torch = None
        real_torch = None
        batchSize = input.shape[0]
        show_nums = min(show_imgs_num, batchSize)

        mean = round(input.min())
        std = round(input.max()) - round(input.min())

        randindex_list = random.sample(list(range(batchSize)), show_nums)
        for randindex in randindex_list:
            input_torch = input[randindex].cpu().detach()
            input_torch = transforms.Normalize([mean, mean, mean], [std, std, std])(
                input_torch)  # (-1,1)=>(0,1)   mean = -1,std = 2

            prediction_torch = fake[randindex].cpu().detach()
            prediction_torch = transforms.Normalize([mean, mean, mean], [std, std, std])(prediction_torch)

            real_torch = real[randindex].cpu().detach()
            real_torch = transforms.Normalize([mean, mean, mean], [std, std, std])(real_torch)
            out_1 = torch.stack((input_torch, prediction_torch, real_torch))
            if out is None:
                out = out_1
            else:
                out = torch.cat((out_1, out))
        out = make_grid(out, nrow=3)
        self.writer.add_image('%s-in-pred-real' % tag, out, global_step)

        input = transforms.ToPILImage()(input_torch).convert(mode)
        prediction = transforms.ToPILImage()(prediction_torch).convert(mode)
        real = transforms.ToPILImage()(real_torch).convert(mode)

        buildDir("plots")
        in_filename = "plots/%s/E%03d_in_.png" % (tag, global_step)
        real_filename = "plots/%s/E%03d_real_.png" % (tag, global_step)
        out_filename = "plots/%s/E%03d_out_.png" % (tag, global_step)
        input.save(in_filename)
        prediction.save(out_filename)
        real.save(real_filename)

    def watchNetwork(self, net, input_shape=None, *input):
        if input_shape is not None:
            assert (isinstance(input_shape, tuple) or isinstance(input_shape, list)), \
                "param 'input_shape' should be list or tuple."
            input = torch.autograd.Variable(torch.Tensor(input_shape), requires_grad=True)
            res = net(input)
        else:
            res = net(*input)
        self.writer.add_graph(net, res)

    def close(self):
        self.writer.close()


def buildDir(dirs=("plots", "plots/Test", "plots/Train", "plots/Valid", "checkpoint")):
    for dir in dirs:
        if not os.path.exists(dir):
            print("%s directory is not found. Build now!" % dir)
            os.mkdir(dir)
