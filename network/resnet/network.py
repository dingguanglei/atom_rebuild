import torch
import torch.nn as nn
from torch.nn import Sequential, Conv2d, Dropout, LeakyReLU, Upsample, BatchNorm2d, Sigmoid, Tanh, Linear
from torch.autograd import Variable
import numpy as np


def weights_init(m):
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight)
        m.bias.data.zero_()
    elif isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight)
        m.bias.data.zero_()
    elif isinstance(m, nn.ConvTranspose2d):
        nn.init.kaiming_normal_(m.weight)
        m.bias.data.zero_()
    elif isinstance(m, nn.InstanceNorm2d):
        nn.init.normal_(m.weight, 1.0, 0.02)
        m.bias.data.fill_(0)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.normal_(m.weight, 1.0, 0.02)
        m.bias.data.fill_(0)
    else:
        pass


def get_norm_layer(norm_type):
    if norm_type == 'batch':
        norm_layer = nn.BatchNorm2d
    elif norm_type == 'instance':
        norm_layer = nn.InstanceNorm2d
    else:
        print('normalization layer [%s] is not found' % norm_type)
    return norm_layer


def define_G(input_nc, output_nc, ngf, norm='instance', use_dropout=False, gpu_ids=[], n_blocks=32):
    netG = None
    use_gpu = len(gpu_ids) > 0
    norm_layer = get_norm_layer(norm_type=norm)

    if use_gpu:
        assert (torch.cuda.is_available())

    netG = ResnetGenerator(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=n_blocks,
                           gpu_ids=gpu_ids)

    if len(gpu_ids) > 0:
        netG.cuda(gpu_ids[0])
    netG.apply(weights_init)
    return netG


def define_D(input_nc, ndf, norm='instance', use_sigmoid=False, gpu_ids=[], n_layers=4):
    netD = None
    use_gpu = len(gpu_ids) > 0
    norm_layer = get_norm_layer(norm_type=norm)

    if use_gpu:
        assert (torch.cuda.is_available())

    netD = NLayerDiscriminator(input_nc, ndf, n_layers=n_layers, norm_layer=norm_layer, use_sigmoid=use_sigmoid,
                               gpu_ids=gpu_ids)

    if use_gpu:
        netD.cuda(gpu_ids[0])
    netD.apply(weights_init)
    return netD


def print_network(net):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print(net)
    print('Total number of parameters: %d' % num_params)


# Defines the GAN loss which uses either LSGAN or the regular GAN.
class GANLoss(nn.Module):
    def __init__(self, use_lsgan=True, target_real_label=1.0, target_fake_label=0.0,
                 tensor=torch.FloatTensor):
        super(GANLoss, self).__init__()
        self.real_label = target_real_label
        self.fake_label = target_fake_label
        self.real_label_var = None
        self.fake_label_var = None
        self.Tensor = tensor

        if use_lsgan:
            self.loss = nn.MSELoss()
        else:
            self.loss = nn.BCELoss()

    def get_target_tensor(self, input, target_is_real):
        target_tensor = None
        if target_is_real:
            create_label = ((self.real_label_var is None) or
                            (self.real_label_var.numel() != input.numel()))
            if create_label:
                real_tensor = self.Tensor(input.size()).fill_(self.real_label)
                self.real_label_var = Variable(real_tensor, requires_grad=False)
            target_tensor = self.real_label_var
        else:
            create_label = ((self.fake_label_var is None) or
                            (self.fake_label_var.numel() != input.numel()))
            if create_label:
                fake_tensor = self.Tensor(input.size()).fill_(self.fake_label)
                self.fake_label_var = Variable(fake_tensor, requires_grad=False)
            target_tensor = self.fake_label_var
        return target_tensor

    def __call__(self, input, target_is_real):
        target_tensor = self.get_target_tensor(input, target_is_real)
        return self.loss(input, target_tensor.cuda())


# Defines the generator that consists of Resnet blocks between a few
# downsampling/upsampling operations.
class ResnetGenerator(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.InstanceNorm2d, use_dropout=False, n_blocks=6,
                 gpu_ids=[]):
        assert (n_blocks >= 0 and n_blocks % 2 == 0)
        super(ResnetGenerator, self).__init__()
        self.input_nc = input_nc
        self.output_nc = output_nc
        self.ngf = ngf
        self.gpu_ids = gpu_ids

        model = [nn.Conv2d(input_nc, ngf, kernel_size=7, padding=3), nn.ReLU()]  # ngf,256,256
        pre_mult = 1
        n_downsampling = int(n_blocks / 2)
        for i in range(n_downsampling):
            mult = 2 ** i
            model += [nn.Conv2d(ngf * pre_mult, ngf * mult, kernel_size=4, stride=2, padding=1),  # 1/2
                      norm_layer(int(ngf * mult), affine=True),
                      nn.ReLU()]
            model += [
                ResnetBlock(ngf * mult, 'zero', norm_layer=norm_layer, use_dropout=use_dropout)]  # ngf * 2 * 2,128,128
            pre_mult = mult

        model += [ResnetBlock(ngf * pre_mult, 'zero', norm_layer=norm_layer, use_dropout=use_dropout)]

        for i in range(n_downsampling):
            mult = 2 ** (n_downsampling - i - 1)
            model += [
                ResnetBlock(ngf * mult, 'zero', norm_layer=norm_layer, use_dropout=use_dropout)]  # ngf * 2 * 2,128,128
            model += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2),
                                         kernel_size=3, stride=2,
                                         padding=1, output_padding=1),
                      norm_layer(int(ngf * mult / 2), affine=True),
                      nn.ReLU()]

        model += [nn.Conv2d(ngf // 2, output_nc, kernel_size=7, padding=3, stride=1)]
        model += [nn.Tanh()]

        self.model = nn.Sequential(*model)

    def forward(self, input):
        if self.gpu_ids and isinstance(input.data, torch.cuda.FloatTensor):
            return nn.parallel.data_parallel(self.model, input, self.gpu_ids)
        else:
            return self.model(input)


# Define a resnet block
class ResnetBlock(nn.Module):
    def __init__(self, dim, padding_type, norm_layer, use_dropout):
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, use_dropout)

    def build_conv_block(self, dim, padding_type, norm_layer, use_dropout):
        conv_block = []
        p = 0
        assert (padding_type == 'zero')
        p = 1

        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p),
                       norm_layer(dim, affine=True),
                       nn.ReLU(True)]
        if use_dropout:
            conv_block += [nn.Dropout(use_dropout)]
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p),
                       norm_layer(dim, affine=True)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        out = x + self.conv_block(x)
        return out


# Defines the PatchGAN discriminator.
class NLayerDiscriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.InstanceNorm2d, use_sigmoid=False, gpu_ids=[]):
        super(NLayerDiscriminator, self).__init__()
        self.gpu_ids = gpu_ids

        kw = 4
        # padw = int(np.ceil((kw - 1) / 2))
        padw = 1
        sequence = [
            nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw),
            nn.LeakyReLU(0.2, True)
        ]

        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2,
                          padding=padw), norm_layer(ndf * nf_mult,
                                                    affine=True), nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=4, stride=2,
                      padding=1), norm_layer(ndf * nf_mult,
                                             affine=True), nn.LeakyReLU(0.2, True)
        ]

        sequence += [nn.Conv2d(ndf * nf_mult, ndf * nf_mult, kernel_size=4, stride=1, padding=0, groups=ndf * nf_mult)]
        sequence += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=1, stride=1, padding=0)]

        if use_sigmoid:
            sequence += [nn.Sigmoid()]

        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        if len(self.gpu_ids) and isinstance(input.data, torch.cuda.FloatTensor):
            return nn.parallel.data_parallel(self.model, input, self.gpu_ids)
        else:
            return self.model(input)


# ----------------------------------------------
class Discriminator(nn.Module):
    def __init__(self, input_nc=1, output_nc=1, depth=64, dropout=0.5,use_sigmoid= True):
        super(Discriminator, self).__init__()
        self.use_sigmoid = use_sigmoid
        # 256 x 256
        self.layer1 = Sequential(Conv2d(input_nc + output_nc, depth, kernel_size=8, stride=2, padding=3),
                                 LeakyReLU(0.2))
        # 128 x 128
        self.layer2 = Sequential(Conv2d(depth, depth * 2, kernel_size=4, stride=2, padding=1),
                                 BatchNorm2d(depth * 2),
                                 LeakyReLU(0.2))
        # 64 x 64
        self.layer3 = Sequential(Conv2d(depth * 2, depth * 4, kernel_size=4, stride=2, padding=1),
                                 BatchNorm2d(depth * 4),
                                 LeakyReLU(0.2))
        # 32 x 32
        self.layer4 = Sequential(Conv2d(depth * 4, depth * 8, kernel_size=4, stride=2, padding=1),
                                 BatchNorm2d(depth * 8),
                                 LeakyReLU(0.2))
        # 16 x 16
        self.layer5 = Sequential(Conv2d(depth * 8, output_nc, kernel_size=1, stride=1, padding=0),
                                 LeakyReLU(0.2))
        # 16 x 16 ,1
        self.liner = Linear(256, 1)
        self.sigmoid = Sigmoid()

    def forward(self, x, g):
        out = torch.cat((x, g), 1)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out).view(out.size(0), -1)
        if self.use_sigmoid:
            out = self.sigmoid(self.liner(out))
        else:
            out = self.liner(out)
        return out

    def save(self):
        torch.save(self.state_dict(), 'model/D.pkl')

    def load(self):
        self.load_state_dict(torch.load('model/D.pkl'))


def define_Unet(input_nc, output_nc, ngf, gpu_ids=[]):
    netG = UnetGenerator(input_nc, output_nc, ngf)
    use_gpu = len(gpu_ids) > 0

    if use_gpu:
        assert (torch.cuda.is_available())
    if len(gpu_ids) > 0:
        netG.cuda(gpu_ids[0])
    netG = torch.nn.DataParallel(netG, gpu_ids)
    netG.apply(weights_init)
    return netG


def define_dis(input_nc, output_nc, ngf, gpu_ids=[],use_sigmoid = True):
    netD = Discriminator(input_nc, output_nc, ngf,use_sigmoid=use_sigmoid)
    use_gpu = len(gpu_ids) > 0

    if use_gpu:
        assert (torch.cuda.is_available())
    if len(gpu_ids) > 0:
        netD.cuda(gpu_ids[0])
    netD = torch.nn.DataParallel(netD, gpu_ids)
    netD.apply(weights_init)
    return netD


class UnetGenerator(nn.Module):
    def __init__(self, input_nc=1, output_nc=1, depth=32, momentum=0.8,dropout = 0):
        super(UnetGenerator, self).__init__()
        '''
        # 256 x 256
        self.e1 = nn.Sequential(nn.Conv2d(input_nc,depth,kernel_size=4,stride=2,padding=1),
                                 nn.LeakyReLU(0.2, inplace=True))
        # 128 x 128
        self.e2 = nn.Sequential(nn.Conv2d(depth,depth*2,kernel_size=4,stride=2,padding=1),
                                 nn.BatchNorm2d(depth*2),
                                 nn.LeakyReLU(0.2, inplace=True))
        # 64 x 64
        self.e3 = nn.Sequential(nn.Conv2d(depth*2,depth*4,kernel_size=4,stride=2,padding=1),
                                 nn.BatchNorm2d(depth*4),
                                 nn.LeakyReLU(0.2, inplace=True))
        # 32 x 32
        self.e4 = nn.Sequential(nn.Conv2d(depth*4,depth*8,kernel_size=4,stride=2,padding=1),
                                 nn.BatchNorm2d(depth*8),
                                 nn.LeakyReLU(0.2, inplace=True))
        # 16 x 16
        self.e5 = nn.Sequential(nn.Conv2d(depth*8,depth*8,kernel_size=4,stride=2,padding=1),
                                 nn.BatchNorm2d(depth*8),
                                 nn.LeakyReLU(0.2, inplace=True))
        # 8 x 8
        self.e6 = nn.Sequential(nn.Conv2d(depth*8,depth*8,kernel_size=4,stride=2,padding=1),
                                 nn.BatchNorm2d(depth*8),
                                 nn.LeakyReLU(0.2, inplace=True))
        # 4 x 4
        self.e7 = nn.Sequential(nn.Conv2d(depth*8,depth*8,kernel_size=4,stride=2,padding=1),
                                 nn.BatchNorm2d(depth*8),
                                 nn.LeakyReLU(0.2, inplace=True))
        # 2 x 2
        self.e8 = nn.Sequential(nn.Conv2d(depth*8,depth*8,kernel_size=4,stride=2,padding=1),
                                 nn.LeakyReLU(0.2, inplace=True))
        # 1 x 1
        self.d1 = nn.Sequential(nn.ConvTranspose2d(depth*8,depth*8,kernel_size=4,stride=2,padding=1),
                                nn.BatchNorm2d(depth*8),
                                nn.Dropout())
        # 2 x 2
        self.d2 = nn.Sequential(nn.ConvTranspose2d(depth*8*2,depth*8,kernel_size=4,stride=2,padding=1),
                                nn.BatchNorm2d(depth*8),
                                nn.Dropout())
        # 4 x 4
        self.d3 = nn.Sequential(nn.ConvTranspose2d(depth*8*2,depth*8,kernel_size=4,stride=2,padding=1),
                                nn.BatchNorm2d(depth*8),
                                nn.Dropout())
        # 8 x 8
        self.d4 = nn.Sequential(nn.ConvTranspose2d(depth*8*2,depth*8,kernel_size=4,stride=2,padding=1),
                                nn.BatchNorm2d(depth*8))
        # 16 x 16
        self.d5 = nn.Sequential(nn.ConvTranspose2d(depth*8*2,depth*4,kernel_size=4,stride=2,padding=1),
                                nn.BatchNorm2d(depth*4))
        # 32 x 32
        self.d6 = nn.Sequential(nn.ConvTranspose2d(depth*4*2,depth*2,kernel_size=4,stride=2,padding=1),
                                nn.BatchNorm2d(depth*2))
        # 64 x 64
        self.d7 = nn.Sequential(nn.ConvTranspose2d(depth*2*2,depth,kernel_size=4,stride=2,padding=1),
                                nn.BatchNorm2d(depth))
        # 128 x 128
        self.d8 = nn.ConvTranspose2d(depth*2,output_nc,kernel_size=4,stride=2,padding=1)
        # 256 x 256
        self.relu = nn.ReLU(inplace=True)
        self.tanh = nn.Tanh()
    def forward(self,x):
        # encoder
        out_e1 = self.e1(x)              # 128 x 128
        out_e2 = self.e2(out_e1)             # 64 x 64
        out_e3 = self.e3(out_e2)             # 32 x 32
        out_e4 = self.e4(out_e3)             # 16 x 16
        out_e5 = self.e5(out_e4)             # 8 x 8
        out_e6 = self.e6(out_e5)             # 4 x 4
        out_e7 = self.e7(out_e6)             # 2 x 2
        out_e8 = self.e8(out_e7)             # 1 x 1
        # decoder
        out_d1 = self.d1(self.relu(out_e8))  # 2 x 2
        out_d1_ = torch.cat((out_d1, out_e7),1)
        out_d2 = self.d2(self.relu(out_d1_)) # 4 x 4
        out_d2_ = torch.cat((out_d2, out_e6),1)
        out_d3 = self.d3(self.relu(out_d2_)) # 8 x 8
        out_d3_ = torch.cat((out_d3, out_e5),1)
        out_d4 = self.d4(self.relu(out_d3_)) # 16 x 16
        out_d4_ = torch.cat((out_d4, out_e4),1)
        out_d5 = self.d5(self.relu(out_d4_)) # 32 x 32
        out_d5_ = torch.cat((out_d5, out_e3),1)
        out_d6 = self.d6(self.relu(out_d5_)) # 64 x 64
        out_d6_ = torch.cat((out_d6, out_e2),1)
        out_d7 = self.d7(self.relu(out_d6_)) # 128 x 128
        out_d7_ = torch.cat((out_d7, out_e1),1)
        out_d8 = self.d8(self.relu(out_d7_)) # 256 x 256
        out = self.tanh(out_d8)
        return out
        '''
        # input_img = Input(shape=self.img_shape)  # 256, 256, 1
        # # Downsampling
        # d0 = conv2d(input_img, depth * 1, bn=False, stride=1, name="d0")  # 256, 256, 16
        # d0 = conv2d(d0, depth * 1, bn=False, f_size=3, stride=1, name="d0_1")  # 256, 256, 16
        # d1 = conv2d(d0, depth * 2, bn=False, name="d1")  # 128, 128, 32
        # d2 = conv2d(d1, depth * 2, name="d2")  # 64, 64, 32
        # d2 = conv2d(d2, depth * 2, f_size=3, stride=1, name="d2_1")  # 64, 64, 32
        # d3 = conv2d(d2, depth * 4, name="d3")  # 32, 32, 64
        # d4 = conv2d(d3, depth * 4, name="d4")  # 16, 16, 64
        # d4 = conv2d(d4, depth * 4, f_size=3, stride=1, name="d4_1")  # 16, 16, 64
        # d5 = conv2d(d4, depth * 8, name="d5")  # 8, 8, 128
        # d6 = conv2d(d5, depth * 8, name="d6")  # 4, 4, 128
        # d6 = conv2d(d6, depth * 8, f_size=3, stride=1, name="d6_1")  # 4, 4, 128
        # d7 = conv2d(d6, depth * 16, name="d7")  # 2, 2, 256
        # d8 = conv2d(d7, depth * 16, f_size=2, stride=1, padding="valid", name="d8")
        self.depth = depth
        self.d0 = Sequential(Conv2d(input_nc, depth, 7, 1, 3), LeakyReLU(0.2),
                             Conv2d(depth, depth, 5, 1, 2), LeakyReLU(0.2))  # 256, 256, 16
        self.d1 = Sequential(Conv2d(depth, depth * 2, 4, 2, 1), LeakyReLU(0.2),
                             Conv2d(depth * 2, depth * 2, 3, 1, 1),
                             BatchNorm2d(depth * 2, momentum=momentum), LeakyReLU(0.2))  # 128, 128, 32
        self.d2 = Sequential(Conv2d(depth * 2, depth * 2, 4, 2, 1),
                             BatchNorm2d(depth * 2, momentum=momentum), LeakyReLU(0.2),
                             Conv2d(depth * 2, depth * 2, 3, 1, 1),
                             BatchNorm2d(depth * 2, momentum=momentum), LeakyReLU(0.2))  # 64, 64, 32
        self.d3 = Sequential(Conv2d(depth * 2, depth * 4, 4, 2, 1),
                             BatchNorm2d(depth * 4, momentum=momentum), LeakyReLU(0.2),
                             Conv2d(depth * 4, depth * 4, 3, 1, 1),
                             BatchNorm2d(depth * 4, momentum=momentum), LeakyReLU(0.2))  # 32, 32, 64
        self.d4 = Sequential(Conv2d(depth * 4, depth * 4, 4, 2, 1),
                             BatchNorm2d(depth * 4, momentum=momentum), LeakyReLU(0.2),
                             Conv2d(depth * 4, depth * 4, 3, 1, 1),
                             BatchNorm2d(depth * 4, momentum=momentum), LeakyReLU(0.2))  # 16, 16, 64
        self.d5 = Sequential(Conv2d(depth * 4, depth * 8, 4, 2, 1),
                             BatchNorm2d(depth * 8, momentum=momentum), LeakyReLU(0.2),
                             Conv2d(depth * 8, depth * 8, 3, 1, 1),
                             BatchNorm2d(depth * 8, momentum=momentum), LeakyReLU(0.2))  # 8, 8, 128
        self.d6 = Sequential(Conv2d(depth * 8, depth * 8, 4, 2, 1),
                             BatchNorm2d(depth * 8, momentum=momentum), LeakyReLU(0.2),
                             Conv2d(depth * 8, depth * 8, 3, 1, 1),
                             BatchNorm2d(depth * 8, momentum=momentum), LeakyReLU(0.2))  # 4, 4, 128
        self.d7 = Sequential(Conv2d(depth * 8, depth * 16, 4, 2, 1),
                             BatchNorm2d(depth * 16, momentum=momentum), LeakyReLU(0.2),
                             Conv2d(depth * 16, depth * 16, 3, 1, 1),
                             BatchNorm2d(depth * 16, momentum=momentum), LeakyReLU(0.2))  # 2, 2, 256
        self.d8 = Sequential(Conv2d(depth * 16, depth * 16, 4, 2, 1),
                             BatchNorm2d(depth * 16, momentum=momentum), LeakyReLU(0.2),
                             Conv2d(depth * 16, depth * 16, 3, 1, 1),
                             BatchNorm2d(depth * 16, momentum=momentum), LeakyReLU(0.2),
                             )  # 2, 2, 256

        # # Upsampling
        # u0 = deconv2d(d8, d7, depth * 16)  # 2, 2, depth * 16  + depth * 16，depth * 8
        # u1 = deconv2d(u0, d6, depth * 8)  # 4, 4, depth * 8  + depth * 8，depth * 8
        # u2 = deconv2d(u1, d5, depth * 8)  # 8 ,8, depth * 8 + depth * 8,depth * 4
        # u3 = deconv2d(u2, d4, depth * 4)  # 16,16,depth * 4 + depth * 4，depth * 4
        # u4 = deconv2d(u3, d3, depth * 4)  # 32,32,depth * 4 + depth * 4，depth * 2
        # u5 = deconv2d(u4, d2, depth * 2)  # 64,64,depth * 2 + depth * 2，depth * 2
        # u6 = deconv2d(u5, d1, depth * 2)  # 128,128,depth * 2 + depth * 2，depth * 2
        # u7 = deconv2d(u6, d0, depth * 1)  # 256,256,depth * 2 + depth * 2, depth * 1
        # # u8 = UpSampling2D(size=2)(u7)  #256,256,depth * 2 + depth * 2，depth * 2
        # u8 = Conv2D(self.channels, kernel_size=5, strides=1, padding='same', activation=None)(u7)  # 256,256,1
        # output_img = Activation(sigmoid10)(u8)
        self.u0 = Sequential(Upsample(scale_factor=2),
                             Conv2d(depth * 16, depth * 16, 3, 1, 1), BatchNorm2d(depth * 16, momentum=momentum), LeakyReLU(0.2),
                             Dropout(dropout, True))
        self.u1 = Sequential(Upsample(scale_factor=2),
                             Conv2d(depth * 8, depth * 8, 3, 1, 1), BatchNorm2d(depth * 8, momentum=momentum), LeakyReLU(0.2),
                             Dropout(dropout, True))
        self.u2 = Sequential(Upsample(scale_factor=2),
                             Conv2d(depth * 4, depth * 4, 3, 1, 1) ,BatchNorm2d(depth * 4, momentum=momentum), LeakyReLU(0.2),
                             Dropout(dropout, True))
        self.u3 = Sequential(Upsample(scale_factor=2),
                             Conv2d(depth * 4, depth * 4, 3, 1, 1), BatchNorm2d(depth * 4, momentum=momentum), LeakyReLU(0.2),
                             Dropout(dropout, True))
        self.u4 = Sequential(Upsample(scale_factor=2),
                             Conv2d(depth * 2, depth * 2, 3, 1, 1), BatchNorm2d(depth * 2, momentum=momentum), LeakyReLU(0.2),
                             Dropout(dropout, True))
        self.u5 = Sequential(Upsample(scale_factor=2),
                             Conv2d(depth * 2, depth * 2, 3, 1, 1), BatchNorm2d(depth * 2, momentum=momentum), LeakyReLU(0.2),
                             Dropout(dropout, True))
        self.u6 = Sequential(Upsample(scale_factor=2),
                             Conv2d(depth * 1, depth * 1, 3, 1, 1), BatchNorm2d(depth * 1, momentum=momentum), LeakyReLU(0.2),
                             Dropout(dropout, True))
        # self.u7 = Sequential(Upsample(scale_factor=2),
        #                      Conv2d(depth * 1, depth * 1, 3, 1, 1), LeakyReLU(0.2),
        #                      Dropout(0.1), BatchNorm2d(depth * 1, momentum=momentum))
        self.u7 = Sequential(Conv2d(depth * 1, output_nc, 5, 1, 2), Tanh())
        # self.conv1 = Conv2d(input_nc, depth, 4, 2, 1)
        # self.conv2 = Conv2d(depth, depth * 2, 4, 2, 1)
        # self.conv3 = Conv2d(depth * 2, depth * 4, 4, 2, 1)
        # self.conv4 = Conv2d(depth * 4, depth * 8, 4, 2, 1)
        # self.conv5 = Conv2d(depth * 8, depth * 8, 4, 2, 1)
        # self.conv6 = Conv2d(depth * 8, depth * 8, 4, 2, 1)
        # self.conv7 = Conv2d(depth * 8, depth * 8, 4, 2, 1)
        # self.conv8 = Conv2d(depth * 8, depth * 8, 4, 2, 1)

        # self.dconv1 = ConvTranspose2d(depth * 8, depth * 8, 4, 2, 1)
        # self.dconv2 = ConvTranspose2d(depth * 8 * 2, depth * 8, 4, 2, 1)
        # self.dconv3 = ConvTranspose2d(depth * 8 * 2, depth * 8, 4, 2, 1)
        # self.dconv4 = ConvTranspose2d(depth * 8 * 2, depth * 8, 4, 2, 1)
        # self.dconv5 = ConvTranspose2d(depth * 8 * 2, depth * 4, 4, 2, 1)
        # self.dconv6 = ConvTranspose2d(depth * 4 * 2, depth * 2, 4, 2, 1)
        # self.dconv7 = ConvTranspose2d(depth * 2 * 2, depth, 4, 2, 1)
        # self.dconv8 = ConvTranspose2d(depth * 2, output_nc, 4, 2, 1)

        # self.batch_norm = BatchNorm2d(depth)
        # self.batch_norm2 = BatchNorm2d(depth * 2)
        # self.batch_norm4 = BatchNorm2d(depth * 4)
        # self.batch_norm8 = BatchNorm2d(depth * 8)
        #
        self.leaky_relu = LeakyReLU(0.2)
        # self.relu = ReLU(True)

        # self.dropout = Dropout(0.2)

        # self.sigmoid = Sigmoid()
        self.conv_32_8 = Conv2d(depth * 16 * 2, depth * 8, 3, 1, 1)
        self.conv_16_8 = Conv2d(depth * 8 * 2, depth * 8, 3, 1, 1)
        self.conv_16_4 = Conv2d(depth * 8 * 2, depth * 4, 3, 1, 1)
        self.conv_8_4 = Conv2d(depth * 4 * 2, depth * 4, 3, 1, 1)
        self.conv_8_2 = Conv2d(depth * 4 * 2, depth * 2, 3, 1, 1)
        self.conv_4_2 = Conv2d(depth * 2 * 2, depth * 2, 3, 1, 1)
        self.conv_4_1 = Conv2d(depth * 2 * 2, depth * 1, 3, 1, 1)
        self.conv_2_1 = Conv2d(depth * 1 * 2, depth * 1, 3, 1, 1)

    def forward(self, input):
        # Encoder
        # Convolution layers:
        # input is (nc) x 256 x 256
        # e1 = self.conv1(input)
        # # state size is (depth) x 128 x 128
        # e2 = self.batch_norm2(self.conv2(self.relu(e1)))
        # # state size is (depth x 2) x 64 x 64
        # e3 = self.batch_norm4(self.conv3(self.relu(e2)))
        # # state size is (depth x 4) x 32 x 32
        # e4 = self.batch_norm8(self.conv4(self.relu(e3)))
        # # state size is (depth x 8) x 16 x 16
        # e5 = self.batch_norm8(self.conv5(self.relu(e4)))
        # # state size is (depth x 8) x 8 x 8
        # e6 = self.batch_norm8(self.conv6(self.relu(e5)))
        # # state size is (depth x 8) x 4 x 4
        # e7 = self.batch_norm8(self.conv7(self.relu(e6)))
        # # state size is (depth x 8) x 2 x 2
        # # No batch norm on output of Encoder
        # e8 = self.conv8(self.relu(e7))
        #
        # # Decoder
        # # Deconvolution layers:
        # # state size is (depth x 8) x 1 x 1
        # # d1_ = self.dropout(self.batch_norm8(self.dconv1(self.leaky_relu(e8))))
        # d1_ = self.batch_norm8(self.dconv1(self.relu(e8)))
        #
        # # state size is (depth x 8) x 2 x 2
        # d1 = torch.cat((d1_, e7), 1)
        # # d2_ = self.dropout(self.batch_norm8(self.dconv2(self.leaky_relu(d1))))
        # d2_ = self.batch_norm8(self.dconv2(self.relu(d1)))
        # # state size is (depth x 8) x 4 x 4
        # d2 = torch.cat((d2_, e6), 1)
        # # d3_ = self.dropout(self.batch_norm8(self.dconv3(self.leaky_relu(d2))))
        # d3_ = self.batch_norm8(self.dconv3(self.relu(d2)))
        #
        # # state size is (depth x 8) x 8 x 8
        # d3 = torch.cat((d3_, e5), 1)
        # d4_ = self.batch_norm8(self.dconv4(self.relu(d3)))
        # # state size is (depth x 8) x 16 x 16
        # d4 = torch.cat((d4_, e4), 1)
        # d5_ = self.batch_norm4(self.dconv5(self.relu(d4)))
        # # state size is (depth x 4) x 32 x 32
        # d5 = torch.cat((d5_, e3), 1)
        # d6_ = self.batch_norm2(self.dconv6(self.relu(d5)))
        # # state size is (depth x 2) x 64 x 64
        # d6 = torch.cat((d6_, e2), 1)
        # d7_ = self.batch_norm(self.dconv7(self.relu(d6)))
        # # state size is (depth) x 128 x 128
        # d7 = torch.cat((d7_, e1), 1)
        # d8 = self.dconv8(self.relu(d7))
        # # state size is (nc) x 256 x 256
        # output = self.sigmoid(d8)
        d0 = self.d0(input)  # 1,256,256
        d1 = self.d1(d0)  # 2,128,128
        d2 = self.d2(d1)  # 2,64,64
        d3 = self.d3(d2)  # 4,32,32
        d4 = self.d4(d3)  # 4,16,16
        d5 = self.d5(d4)  # 8,8,8
        d6 = self.d6(d5)  # 8,4,4
        d7 = self.d7(d6)  # 16,2,2
        d8 = self.d8(d7)  # 16,1,1
        depth = self.depth

        # depth * 16, depth * 16                  depth * 8
        d8 = self.u0(d8)
        u0 = torch.cat((d8, d7), 1)  # 32, 2, 2
        u0 = self.leaky_relu(self.conv_32_8(u0))  # 8, 2 ,2

        # depth * 8, depth * 8                    depth * 8
        u0 = self.u1(u0)
        u1 = torch.cat((u0, d6), 1)
        u1 = self.leaky_relu(self.conv_16_8(u1))

        # depth * 8, depth * 8                    depth * 4
        u1 = self.u1(u1)
        u2 = torch.cat((u1, d5), 1)
        u2 = self.leaky_relu(self.conv_16_4(u2))

        # depth * 4 , depth * 4                   depth * 2
        u2 = self.u2(u2)
        u3 = torch.cat((u2, d4), 1)
        u3 = self.leaky_relu(self.conv_8_4(u3))

        # depth * 4 , depth * 4                   depth * 2
        u3 = self.u3(u3)
        u4 = torch.cat((u3, d3), 1)
        u4 = self.leaky_relu(self.conv_8_2(u4))

        # depth * 2 , depth * 2                   depth * 1
        u4 = self.u4(u4)
        u5 = torch.cat((u4, d2), 1)
        u5 = self.leaky_relu(self.conv_4_2(u5))

        # depth * 2 , depth * 2                   depth * 1
        u5 = self.u5(u5)
        u6 = torch.cat((u5, d1), 1)
        u6 = self.leaky_relu(self.conv_4_1(u6))

        # depth * 1 , depth * 1                   depth * 1
        u6 = self.u6(u6)
        u7 = torch.cat((u6, d0), 1)
        u7 = self.leaky_relu(self.conv_2_1(u7))
        output = self.u7(u7)
        # # Upsampling
        # u0 = deconv2d(d8, d7, depth * 16)  # 2, 2, depth * 16  + depth * 16，depth * 8
        # u1 = deconv2d(u0, d6, depth * 8)  # 4, 4, depth * 8  + depth * 8，depth * 8
        # u2 = deconv2d(u1, d5, depth * 8)  # 8 ,8, depth * 8 + depth * 8,depth * 4
        # u3 = deconv2d(u2, d4, depth * 4)  # 16,16,depth * 4 + depth * 4，depth * 4
        # u4 = deconv2d(u3, d3, depth * 4)  # 32,32,depth * 4 + depth * 4，depth * 2
        # u5 = deconv2d(u4, d2, depth * 2)  # 64,64,depth * 2 + depth * 2，depth * 2
        # u6 = deconv2d(u5, d1, depth * 2)  # 128,128,depth * 2 + depth * 2，depth * 2
        # u7 = deconv2d(u6, d0, depth * 1)  # 256,256,depth * 2 + depth * 2, depth * 1
        # # u8 = UpSampling2D(size=2)(u7)  #256,256,depth * 2 + depth * 2，depth * 2
        # u8 = Conv2D(self.channels, kernel_size=5, strides=1, padding='same', activation=None)(u7)  # 256,256,1
        # output_img = Activation(sigmoid10)(u8)

        return output

    def save(self):
        torch.save(self.state_dict(), 'model/G.pkl')

    def load(self):
        self.load_state_dict(torch.load('model/G.pkl'))
