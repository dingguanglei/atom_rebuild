# coding=utf-8
from torch.nn import Module, Sequential, Conv2d, BatchNorm2d, InstanceNorm2d, LeakyReLU, Linear, Sigmoid, Upsample, \
    Tanh, Dropout, ReLU, MaxPool2d, AvgPool2d, ModuleList
import torch
from .shared.basic import *


class Wnet_G(Module):
    def __init__(self, input_nc=1, output_nc=1, depth=32, norm_type=None,
                 active_type="ReLU", branches=3):
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
            if i >= 2:
                groups = 1
            else:
                groups = branches
            self.add_module("upsample_block_" + str(i + 1), upsampleBlock((self._depth(i + 2), self._depth(i)),
                                                                          (3, 1, 1),
                                                                          active_type=active_type,
                                                                          norm_type=norm_type, groups=groups))
            self.upsample += ["upsample_block_" + str(i + 1)]

        # output bloack
        self.output = normalBlock((depth, branches), (7, 1, 3), active_type="Tanh", ksp_2=(5, 1, 2), groups=branches)

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


class Branch_NLayer_D(Module):
    def __init__(self, input_nc=1, output_nc=3, depth=64, use_sigmoid=True, norm_type="batch",
                 active_type="ReLU", groups=1):
        super(Branch_NLayer_D, self).__init__()
        self.use_sigmoid = use_sigmoid

        # 256 x 256
        self.layer1 = convLayer(input_nc + output_nc, depth, 8, 2, 3, active_type=active_type, norm_type=None,
                                groups=groups)

        # 128 x 128
        self.layer2 = convLayer(depth, depth * 2, 4, 2, 1, active_type=active_type, norm_type=norm_type,
                                groups=groups)

        # 64 x 64
        self.layer3 = convLayer(depth * 2, depth * 4, 4, 2, 1, active_type=active_type, norm_type=norm_type,
                                groups=groups)

        # 32 x 32
        self.layer4 = convLayer(depth * 4, depth * 8, 4, 2, 1, active_type=active_type, norm_type=norm_type,
                                groups=groups)

        # 16 x 16
        self.layer5 = Conv2d(depth * 8, output_nc, kernel_size=5, stride=1, padding=2)

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

        if self.use_sigmoid:
            out = self.sigmoid(out)

        return out
