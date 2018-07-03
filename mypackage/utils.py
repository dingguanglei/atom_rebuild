# coding=utf-8
from torch import save, load
import torch
import time, os


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


class Model():
    def __init__(self, epoch=30, gpus=None, model_path=None, model_weights_path=None):
        self.epoch = epoch
        self.model_path = "checkpoint/Model_G_{}.pth".format(self.epoch)
        self.model_weights_path = "checkpoint/Model_weights_G_{}.pth".format(self.epoch)
        self.model = self.loadModel(gpus)

    def loadModel(self, gpus=None):
        model_path = self.model_path
        model_weights_path = self.model_weights_path
        if torch.cuda.is_available() and (gpus is not None):
            print("load model uses GPU")
            model = torch.load(model_path, map_location=lambda storage, loc: storage.cuda(gpus)).module
            weights = torch.load(model_weights_path, map_location=lambda storage, loc: storage.cuda(gpus))
            model.load_state_dict(weights)
        else:
            print("load model uses CPU")
            model = torch.load(model_path, map_location=lambda storage, loc: storage).module
            weights = torch.load(model_weights_path, map_location=lambda storage, loc: storage)

            from collections import OrderedDict
            new_state_dict = OrderedDict()
            for k, v in weights.items():
                name = k[7:]  # remove `module.`
                new_state_dict[name] = v
            # load params
            model.load_state_dict(new_state_dict)

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


def loadCheckPoint(netG, netD, epoch, name=""):
    net_g_model_out_path = "checkpoint/{}Model_weights_G_{}.pth".format(name, epoch)
    net_d_model_out_path = "checkpoint/{}Model_weights_D_{}.pth".format(name, epoch)
    netG.load_state_dict(load(net_g_model_out_path))
    netD.load_state_dict(load(net_d_model_out_path))
    print("Checkpoint loaded !")
    return netG, netD


def loadG_Model(epoch):
    model_out_path = "checkpoint/Model_G_{}.pth".format(epoch)
    G_model = load(model_out_path)
    return G_model


def writeNetwork(writer, net, *input):
    if input == None:
        input = torch.autograd.Variable(torch.Tensor(1, 1, 28, 28), requires_grad=True)
    res = net(input)
    writer.add_graph(net, res)
