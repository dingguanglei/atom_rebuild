# coding = utf-8
import torch.cuda
from torch.nn import Conv2d, Linear, ConvTranspose2d, InstanceNorm2d, BatchNorm2d, init, DataParallel


def defineNet(net, gpu_ids=(0, 1, 2, 3), use_weights_init=True):
    print_network(net)
    use_gpu = torch.cuda.is_available()
    model_name= type(net)
    if (len(gpu_ids) == 1) & use_gpu:
        net = net.cuda(gpu_ids[0])
        print("%s model use GPU(%d)!" % (model_name,gpu_ids[0]))
    elif (len(gpu_ids) > 1 )& use_gpu:
        net = DataParallel(net.cuda(), gpu_ids)
        print("%s dataParallel use GPUs%s!" % (model_name,gpu_ids))
    else:
        print("%s model use CPU!"% (model_name))

    if use_weights_init:
        net.apply(weightsInit)
    return net


def print_network(net):
    model_name = type(net)
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    # print(net)
    print('%s Total number of parameters: %d' % (model_name,num_params))


def weightsInit(m):
    if isinstance(m, Conv2d):
        init.kaiming_normal_(m.weight)
        m.bias.data.zero_()
    elif isinstance(m, Linear):
        init.kaiming_normal_(m.weight)
        m.bias.data.zero_()
    elif isinstance(m, ConvTranspose2d):
        init.kaiming_normal_(m.weight)
        m.bias.data.zero_()
    elif isinstance(m, InstanceNorm2d):
        init.normal_(m.weight, 1.0, 0.02)
        m.bias.data.fill_(0)
    elif isinstance(m, BatchNorm2d):
        init.normal_(m.weight, 1.0, 0.02)
        m.bias.data.fill_(0)
    else:
        pass
