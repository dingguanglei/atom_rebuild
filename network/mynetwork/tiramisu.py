from .layers import *
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class FCDenseNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, down_blocks=(5, 5, 5, 5, 5),
                 up_blocks=(5, 5, 5, 5, 5), bottleneck_layers=5,
                 growth_rate=16, out_chans_first_conv=48):
        super(FCDenseNet, self).__init__()
        # super().__init__
        self.down_blocks = down_blocks
        self.up_blocks = up_blocks
        cur_channels_count = 0
        skip_connection_channel_counts = []

        ## First Convolution ##

        self.add_module('firstconv', nn.Conv2d(in_channels=in_channels,
                                               out_channels=out_chans_first_conv, kernel_size=3,
                                               stride=1, padding=1, bias=True))
        cur_channels_count = out_chans_first_conv

        #####################
        # Downsampling path #
        #####################

        self.denseBlocksDown = nn.ModuleList([])
        self.transDownBlocks = nn.ModuleList([])
        for i in range(len(down_blocks)):
            self.denseBlocksDown.append(
                DenseBlock(cur_channels_count, growth_rate, down_blocks[i]))
            cur_channels_count += (growth_rate * down_blocks[i])
            skip_connection_channel_counts.insert(0, cur_channels_count)
            self.transDownBlocks.append(TransitionDown(cur_channels_count))

        #####################
        #     Bottleneck    #
        #####################

        self.add_module('bottleneck', Bottleneck(cur_channels_count,
                                                 growth_rate, bottleneck_layers))
        prev_block_channels = growth_rate * bottleneck_layers
        cur_channels_count += prev_block_channels

        #######################
        #   Upsampling path   #
        #######################

        self.transUpBlocks = nn.ModuleList([])
        self.denseBlocksUp = nn.ModuleList([])
        for i in range(len(up_blocks) - 1):
            self.transUpBlocks.append(TransitionUp(prev_block_channels, prev_block_channels))
            cur_channels_count = prev_block_channels + skip_connection_channel_counts[i]

            self.denseBlocksUp.append(DenseBlock(
                cur_channels_count, growth_rate, up_blocks[i],
                upsample=True))
            prev_block_channels = growth_rate * up_blocks[i]
            cur_channels_count += prev_block_channels

        ## Final DenseBlock ##

        self.transUpBlocks.append(TransitionUp(
            prev_block_channels, prev_block_channels))
        cur_channels_count = prev_block_channels + skip_connection_channel_counts[-1]

        self.denseBlocksUp.append(DenseBlock(
            cur_channels_count, growth_rate, up_blocks[-1],
            upsample=False))
        cur_channels_count += growth_rate * up_blocks[-1]

        ## Softmax ##

        self.finalConv = nn.Conv2d(in_channels=cur_channels_count,
                                   out_channels=out_channels, kernel_size=1, stride=1,
                                   padding=0, bias=True)
        # self.softmax = nn.LogSoftmax(dim=1)
        self.tanh = nn.Tanh()

    def forward(self, x):
        out = self.firstconv(x)

        skip_connections = []
        for i in range(len(self.down_blocks)):
            out = self.denseBlocksDown[i](out)
            skip_connections.append(out)
            out = self.transDownBlocks[i](out)

        out = self.bottleneck(out)
        for i in range(len(self.up_blocks)):
            skip = skip_connections.pop()
            out = self.transUpBlocks[i](out, skip)
            out = self.denseBlocksUp[i](out)

        out = self.finalConv(out)
        # out = self.softmax(out)
        out = self.tanh(out)
        return out


def FCDenseNet57(n_classes, gpu_id=[]):
    if torch.cuda.is_available() and gpu_id:
        model = FCDenseNet(
            in_channels=1, down_blocks=(4, 4, 4, 4, 4),
            up_blocks=(4, 4, 4, 4, 4), bottleneck_layers=4,
            growth_rate=12, out_chans_first_conv=48, n_classes=n_classes)
        model.apply(weights_init)
        return nn.parallel.DataParallel(model, gpu_id)

    else:
        return FCDenseNet(
            in_channels=1, down_blocks=(4, 4, 4, 4, 4),
            up_blocks=(4, 4, 4, 4, 4), bottleneck_layers=4,
            growth_rate=12, out_chans_first_conv=48, n_classes=n_classes)


def FCDenseNet67(n_classes, gpu_id=[]):
    if torch.cuda.is_available() and gpu_id:
        model = FCDenseNet(
            in_channels=1, down_blocks=(5, 5, 5, 5, 5),
            up_blocks=(5, 5, 5, 5, 5), bottleneck_layers=5,
            growth_rate=16, out_chans_first_conv=48, n_classes=n_classes)
        model.apply(weights_init)
        return nn.parallel.DataParallel(model, gpu_id)
    else:

        return FCDenseNet(
            in_channels=1, down_blocks=(5, 5, 5, 5, 5),
            up_blocks=(5, 5, 5, 5, 5), bottleneck_layers=5,
            growth_rate=16, out_chans_first_conv=48, n_classes=n_classes)


def FCDenseNet103(n_classes, gpu_id=[]):
    if torch.cuda.is_available() and gpu_id:
        model = FCDenseNet(
            in_channels=1, down_blocks=(4, 5, 7, 10, 12),
            up_blocks=(12, 10, 7, 5, 4), bottleneck_layers=15,
            growth_rate=16, out_chans_first_conv=48, n_classes=n_classes)
        model.apply(weights_init)
        return nn.parallel.DataParallel(model, gpu_id)
    else:
        return FCDenseNet(
            in_channels=1, down_blocks=(4, 5, 7, 10, 12),
            up_blocks=(12, 10, 7, 5, 4), bottleneck_layers=15,
            growth_rate=16, out_chans_first_conv=48, n_classes=n_classes)


def print_network(net):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print(net)
    print('Total number of parameters: %d' % num_params)


def weights_init(m):
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_uniform_(m.weight)
        m.bias.data.zero_()
    elif isinstance(m, nn.Linear):
        nn.init.kaiming_uniform_(m.weight)
        m.bias.data.zero_()
    elif isinstance(m, nn.ConvTranspose2d):
        nn.init.kaiming_uniform_(m.weight)
        m.bias.data.zero_()
    elif isinstance(m, nn.InstanceNorm2d):
        nn.init.normal_(m.weight, 1.0, 0.02)
        m.bias.data.fill_(0)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.normal_(m.weight, 1.0, 0.02)
        m.bias.data.fill_(0)
    else:
        pass


class FocalLoss(nn.Module):
    r"""
        This criterion is a implemenation of Focal Loss, which is proposed in
        Focal Loss for Dense Object Detection.

            Loss(x, class) = - \alpha (1-softmax(x)[class])^gamma \log(softmax(x)[class])

        The losses are averaged across observations for each minibatch.

        Args:
            alpha(1D Tensor, Variable) : the scalar factor for this criterion
            gamma(float, double) : gamma > 0; reduces the relative loss for well-classiﬁed examples (p > .5),
                                   putting more focus on hard, misclassiﬁed examples
            size_average(bool): By default, the losses are averaged over observations for each minibatch.
                                However, if the field size_average is set to False, the losses are
                                instead summed for each minibatch.


    """

    def __init__(self, class_num, alpha=None, gamma=2, size_average=True):
        super(FocalLoss, self).__init__()
        if alpha is None:
            self.alpha = Variable(torch.ones(class_num, 1))
        else:
            if isinstance(alpha, Variable):
                self.alpha = alpha
            else:
                self.alpha = Variable(alpha)
        self.gamma = gamma
        self.class_num = class_num
        self.size_average = size_average

    def forward(self, inputs, targets):
        N = inputs.size(0)
        C = inputs.size(1)
        P = F.softmax(inputs)

        class_mask = inputs.data.new(N, C).fill_(0)
        class_mask = Variable(class_mask)
        ids = targets.view(-1, 1)
        class_mask.scatter_(1, ids.data, 1.)
        # print(class_mask)

        if inputs.is_cuda and not self.alpha.is_cuda:
            self.alpha = self.alpha.cuda()
        alpha = self.alpha[ids.data.view(-1)]

        probs = (P * class_mask).sum(1).view(-1, 1)

        log_p = probs.log()
        # print('probs size= {}'.format(probs.size()))
        # print(probs)

        batch_loss = -alpha * (torch.pow((1 - probs), self.gamma)) * log_p
        # print('-----bacth_loss------')
        # print(batch_loss)

        if self.size_average:
            loss = batch_loss.mean()
        else:
            loss = batch_loss.sum()
        return loss


class DiceLoss(nn.Module):

    def __init__(self):
        super(DiceLoss, self).__init__()

    def forward(self, inputs, targets):
        loss = self.dice_loss(inputs, targets)

        return loss

    def dice_loss(self, input, target):
        """
        input is a torch variable of size BatchxnclassesxHxW representing log probabilities for each class
        target is a 1-hot representation of the groundtruth, shoud have same size as the input
        """
        assert input.size() == target.size(), "Input sizes must be equal."
        assert input.dim() == 4, "Input must be a 4D Tensor."
        # uniques = np.unique(target.numpy())
        # assert set(list(uniques)) <= set([0, 1]), "target must only contain zeros and ones"

        # probs = F.softmax(input)
        probs = input
        num = probs * target  # b,c,h,w--p*g
        num = torch.sum(num, dim=3)  # b,c,h
        num = torch.sum(num, dim=2)

        den1 = probs * probs  # --p^2
        den1 = torch.sum(den1, dim=3)  # b,c,h
        den1 = torch.sum(den1, dim=2)

        den2 = target * target  # --g^2
        den2 = torch.sum(den2, dim=3)  # b,c,h
        den2 = torch.sum(den2, dim=2)  # b,c

        dice = 2 * (num / (den1 + den2))
        dice_eso = dice[:, 1:]  # we ignore bg dice val, and take the fg

        dice_total = -1 * torch.sum(dice_eso) / dice_eso.size(0)  # divide by batch_sz

        return dice_total
