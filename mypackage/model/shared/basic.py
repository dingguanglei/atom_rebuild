# coding=utf-8
from torch.nn import Module, Sequential, Conv2d, BatchNorm2d, InstanceNorm2d, LeakyReLU, Linear, Sigmoid, Upsample, \
    Tanh, Dropout, ReLU, MaxPool2d, AvgPool2d, ModuleList
import torch

class downsampleBlock(Module):
    def __init__(self, channel_in_out, ksp=(3, 1, 1), pool_type="Max", active_type="ReLU",
                 norm_type=None, groups=1):
        super(downsampleBlock, self).__init__()
        kernel_size, stride, padding = ksp
        c_in, c_out = channel_in_out

        self.convLayer_1 = convLayer(c_in, c_out, kernel_size, stride, padding, active_type, norm_type, groups)

        self.poolLayer = getPoolLayer(pool_type)

        self.convLayer_2 = convLayer(c_out, c_out, kernel_size, stride, padding, active_type, norm_type, groups)

    def forward(self, input):
        out = self.convLayer_1(input)
        out = self.poolLayer(out)
        out = self.convLayer_2(out)
        return out


class upsampleBlock(Module):
    def __init__(self, channel_in_out, ksp=(3, 1, 1), active_type="ReLU", norm_type=None, groups=1):
        super(upsampleBlock, self).__init__()
        kernel_size, stride, padding = ksp
        c_in, c_out = channel_in_out
        c_shrunk = c_in // 2

        self.shrunkConvLayer_1 = convLayer(c_in, c_shrunk, 1, 1, 0, active_type, norm_type, 1)

        self.convLayer_2 = convLayer(c_shrunk, c_out, kernel_size, stride, padding, active_type, norm_type, groups)

        self.upSampleLayer = Upsample(scale_factor=2)

    def forward(self, prev_input, now_input):
        out = torch.cat((prev_input, now_input), 1)
        out = self.shrunkConvLayer_1(out)
        out = self.upSampleLayer(out)
        out = self.convLayer_2(out)
        return out


class convLayer(Module):
    def __init__(self, c_in, c_out, kernel_size=3, stride=1, padding=1, active_type="ReLU", norm_type=None, groups=1):
        super(convLayer, self).__init__()
        norm = getNormLayer(norm_type)
        act = getActiveLayer(active_type)

        self.module_list = ModuleList([])
        self.module_list += [Conv2d(c_in, c_out, kernel_size, stride, padding, groups=groups)]
        if norm:
            self.module_list += [norm(c_out, affine=True)]
        self.module_list += [act]

    def forward(self, input):
        out = input
        for layer in self.module_list:
            out = layer(out)
        return out


class normalBlock(Module):
    def __init__(self, channel_in_out, ksp=(3, 1, 1), active_type="ReLU", norm_type=None, ksp_2=None, groups=1):
        super(normalBlock, self).__init__()
        c_in, c_out = channel_in_out
        kernel_size, stride, padding = ksp

        if ksp_2 is None:
            kernel_size_2, stride_2, padding_2 = ksp
        else:
            kernel_size_2, stride_2, padding_2 = ksp_2

        assert stride + 2 * padding - kernel_size == 0, \
            "kernel_size%s, stride%s, padding%s is not a 'same' group! s+2p-k ==0" % (
                kernel_size, stride, padding)
        assert stride_2 + 2 * padding_2 - kernel_size_2 == 0, \
            "kernel_size%s, stride%s, padding%s is not a 'same' group! s+2p-k ==0" % (
                kernel_size_2, stride_2, padding_2)

        self.convLayer_1 = convLayer(c_in, c_out, kernel_size_2, stride_2, padding_2, active_type, norm_type,
                                     groups=groups)
        self.convLayer_2 = convLayer(c_out, c_out, kernel_size, stride, padding, active_type, norm_type, groups=groups)

    def forward(self, input):
        out = input
        out = self.convLayer_1(out)
        out = self.convLayer_2(out)

        return out


def getNormLayer(norm_type):
    norm_layer = BatchNorm2d
    if norm_type == 'batch':
        norm_layer = BatchNorm2d
    elif norm_type == 'instance':
        norm_layer = InstanceNorm2d
    elif norm_type is None:
        norm_layer = None
    else:
        print('normalization layer [%s] is not found' % norm_type)
    return norm_layer


def getPoolLayer(pool_type):
    pool_layer = MaxPool2d(2, 2)
    if pool_type == 'Max':
        pool_layer = MaxPool2d(2, 2)
    elif pool_type == 'Avg':
        pool_layer = AvgPool2d(2, 2)
    elif pool_type is None:
        pool_layer = None
    else:
        print('pool layer [%s] is not found' % pool_layer)
    return pool_layer


def getActiveLayer(active_type):
    active_layer = ReLU
    if active_type == 'ReLU':
        active_layer = ReLU()
    elif active_type == 'LeakyReLU':
        active_layer = LeakyReLU(0.2)
    elif active_type == 'Tanh':
        active_layer = Tanh()
    elif active_type is None:
        active_layer = None
    else:
        print('active layer [%s] is not found' % active_layer)
    return active_layer
