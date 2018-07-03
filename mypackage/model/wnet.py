# coding=utf-8
from torch.nn import Module, Sequential, Conv2d, BatchNorm2d, InstanceNorm2d, LeakyReLU, Linear, Sigmoid, Upsample, \
    Tanh, Dropout, ReLU, MaxPool2d, AvgPool2d, ModuleList
import torch
import collections


class downsampleBlock(Module):
    def __init__(self, channel_in_out, ksp=(3, 1, 1), pool_type="Max", active_type="ReLU",
                 norm_type=None):
        super(downsampleBlock, self).__init__()
        kernel_size, stride, padding = ksp
        c_in, c_out = channel_in_out

        self.convLayer_1 = _convLayer(c_in, c_out, kernel_size, stride, padding, active_type, norm_type)

        self.poolLayer = getPoolLayer(pool_type)

        self.convLayer_2 = _convLayer(c_out, c_out, kernel_size, stride, padding, active_type, norm_type)

    def forward(self, input):
        out = self.convLayer_1(input)
        out = self.poolLayer(out)
        out = self.convLayer_2(out)
        return out


class upsampleBlock(Module):
    def __init__(self, channel_in_out, ksp=(3, 1, 1), active_type="ReLU", norm_type=None):
        super(upsampleBlock, self).__init__()
        kernel_size, stride, padding = ksp
        c_in, c_out = channel_in_out
        c_shrunk = c_in // 2

        self.shrunkConvLayer_1 = _convLayer(c_in, c_shrunk, 1, 1, 0, active_type, norm_type)

        self.convLayer_2 = _convLayer(c_shrunk, c_out, kernel_size, stride, padding, active_type, norm_type)

        self.upSampleLayer = Upsample(scale_factor=2)

    def forward(self, prev_input, now_input):
        out = torch.cat((prev_input, now_input), 1)
        out = self.shrunkConvLayer_1(out)
        out = self.upSampleLayer(out)
        out = self.convLayer_2(out)
        return out


class _convLayer(Module):
    def __init__(self, c_in, c_out, kernel_size=3, stride=1, padding=1, active_type="ReLU", norm_type=None):
        super(_convLayer, self).__init__()
        norm = getNormLayer(norm_type)
        act = getActiveLayer(active_type)

        self.module_list = ModuleList([])
        self.module_list += [Conv2d(c_in, c_out, kernel_size, stride, padding)]
        if norm:
            self.module_list += [norm(c_out, affine=True)]
        self.module_list += [act]

    def forward(self, input):
        out = input
        for layer in self.module_list:
            out = layer(out)
        return out


class normalBlock(Module):
    def __init__(self, channel_in_out, ksp=(3, 1, 1), active_type="ReLU", norm_type=None, ksp_2=None):
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

        self.convLayer_1 = _convLayer(c_in, c_out, kernel_size, stride, padding, active_type, norm_type)
        self.convLayer_2 = _convLayer(c_out, c_out, kernel_size, stride, padding, active_type, norm_type)

    def forward(self, input):
        out = input
        out = self.convLayer_1(out)
        out = self.convLayer_2(out)

        return out


class Wnet_G(Module):
    def __init__(self, input_nc=1, output_nc=1,depth=32, norm_type=None,
                 active_type="ReLU",branches = 3):
        super(Wnet_G, self).__init__()
        self.depth = depth
        self.branches = branches
        blocks = 6
        max_depth = depth * (2 ** (blocks - 1))
        num_hidden_blocks = blocks - 1

        self.input = normalBlock((input_nc, depth), (7, 1, 3), active_type=active_type,
                                 norm_type=norm_type,
                                 ksp_2=(5, 1, 2))
        # down sample block 2-6, (0,1,2,3,4)
        self.downsample = []
        for i in range(num_hidden_blocks):
            if i >= num_hidden_blocks - 2:
                pool_type = "Avg"
            else:
                pool_type = "Max"
            self.add_module("downsample_block_" + str(i + 1),
                            downsampleBlock((self._depth(i), self._depth(i + 1)),
                                            (3, 1, 1),
                                            pool_type=pool_type,
                                            active_type=active_type))
            self.downsample += ["downsample_block_" + str(i + 1)]

        # bottle neck block
        self.bottleneck = normalBlock((max_depth, max_depth), (3, 1, 1), active_type=active_type,
                                      norm_type=norm_type)

        # up sample block 2-6, (4,3,2,1,0)
        self.upsample = []
        for i in range(num_hidden_blocks - 1, -1, -1):
            self.add_module("upsample_block_" + str(i + 1), upsampleBlock((self._depth(i + 2), self._depth(i)),
                                                                          (3, 1, 1),
                                                                          active_type=active_type,
                                                                          norm_type=norm_type))
            self.upsample += ["upsample_block_" + str(i + 1)]


        # output bloack
        self.output = normalBlock((depth, output_nc), (7, 1, 3), active_type="Tanh", ksp_2=(5, 1, 2))

    def _depth(self, i):
        return self.depth * (2 ** i)

    def forward(self, input):
        d_result = []
        _input = self.input(input)
        d_result.append(_input)
        # down sample block 2-6, (0,1,2,3,4)
        for name in self.downsample:
            _input = self._modules[name](_input)
            d_result.append(_input)

        _input = self.bottleneck(_input)

        # up sample block 2-6, (4,3,2,1,0)
        for name in self.upsample:
            prev_input = d_result.pop()
            _input = self._modules[name](prev_input, _input)
        # output
        out = self.output(_input)

        return out


class Branch_Wnet_G(Module):
    def __init__(self, input_nc=1, output_nc=1, depth=32, norm_type=None,
                 active_type="ReLU", branches=3):
        super(Branch_Wnet_G, self).__init__()
        self.depth = depth
        self.branches = branches
        blocks = 6
        max_depth = depth * (2 ** (blocks - 1))
        num_hidden_blocks = blocks - 1

        self.input = normalBlock((input_nc, depth), (7, 1, 3), active_type=active_type,
                                 norm_type=norm_type,
                                 ksp_2=(5, 1, 2))
        # down sample block 2-6, (0,1,2,3,4)
        self.downsample = []
        for i in range(num_hidden_blocks):
            if i >= num_hidden_blocks - 2:
                pool_type = "Avg"
            else:
                pool_type = "Max"
            self.add_module("downsample_block_" + str(i + 1),
                            downsampleBlock((self._depth(i), self._depth(i + 1)),
                                            (3, 1, 1),
                                            pool_type=pool_type,
                                            active_type=active_type))
            self.downsample += ["downsample_block_" + str(i + 1)]

        # bottle neck block
        self.bottleneck = normalBlock((max_depth, max_depth), (3, 1, 1), active_type=active_type,
                                      norm_type=norm_type)

        # up sample block 2-6, (4,3,2,1,0)
        self.upsample = []
        for i in range(num_hidden_blocks - 1, -1, -1):
            self.add_module("upsample_block_" + str(i + 1), upsampleBlock((self._depth(i + 2), self._depth(i)),
                                                                          (3, 1, 1),
                                                                          active_type=active_type,
                                                                          norm_type=norm_type))
            self.upsample += ["upsample_block_" + str(i + 1)]

        # output bloack
        self.output = normalBlock((depth, output_nc), (7, 1, 3), active_type="Tanh", ksp_2=(5, 1, 2))

    def _depth(self, i):
        return self.depth * (2 ** i)

    def forward(self, input):
        d_result = []
        _input = self.input(input)
        d_result.append(_input)
        # down sample block 2-6, (0,1,2,3,4)
        for name in self.downsample:
            _input = self._modules[name](_input)
            d_result.append(_input)

        _input = self.bottleneck(_input)

        # up sample block 2-6, (4,3,2,1,0)
        for name in self.upsample:
            prev_input = d_result.pop()
            _input = self._modules[name](prev_input, _input)
        # output
        out = self.output(_input)

        return out

class NLayer_D(Module):
    def __init__(self, input_nc=1, output_nc=1, depth=64, use_sigmoid=True, use_liner=True, norm_type="batch",
                 active_type="ReLU"):
        super(NLayer_D, self).__init__()
        self.norm = getNormLayer(norm_type)
        self.active = getActiveLayer(active_type)
        self.use_sigmoid = use_sigmoid
        self.use_liner = use_liner

        # 256 x 256
        self.layer1 = Sequential(Conv2d(input_nc + output_nc, depth, kernel_size=8, stride=2, padding=3),
                                 LeakyReLU(0.2))
        # 128 x 128
        self.layer2 = Sequential(Conv2d(depth, depth * 2, kernel_size=4, stride=2, padding=1),
                                 self.norm(depth * 2, affine=True),
                                 LeakyReLU(0.2))
        # 64 x 64
        self.layer3 = Sequential(Conv2d(depth * 2, depth * 4, kernel_size=4, stride=2, padding=1),
                                 self.norm(depth * 4, affine=True),
                                 LeakyReLU(0.2))
        # 32 x 32
        self.layer4 = Sequential(Conv2d(depth * 4, depth * 8, kernel_size=4, stride=2, padding=1),
                                 self.norm(depth * 8, affine=True),
                                 LeakyReLU(0.2))
        # 16 x 16
        self.layer5 = Sequential(Conv2d(depth * 8, output_nc, kernel_size=5, stride=1, padding=2))
        # 16 x 16 ,1
        self.liner = Linear(256, 1)
        self.sigmoid = Sigmoid()

    def forward(self, x, g):
        out = torch.cat((x, g), 1)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)

        if self.use_liner:
            out = out.view(out.size(0), -1)

        if self.use_sigmoid:
            out = self.sigmoid(out)

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
