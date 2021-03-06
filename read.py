import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
import os
import sys
import argparse
from skimage import data
from torch.autograd import Variable
from mypackage.utils import Model, buildDir
from mypackage.data import TestDataset
from tensorboardX import SummaryWriter

if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
    writer = SummaryWriter("log")
    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch', '-e', type=int, help='display an integer as epoch of model')
    parser.add_argument('--test_dir', '-td', type=str, help='path of test directory', default="test")
    parser.add_argument('--result_dir', '-rd', type=str, help='path of result directory', default="result")
    parser.add_argument('--model_name', '-mn', type=str, help='model name', default="")
    parser.add_argument('--min_size', '-ms', type=str, help='model name', default=32 * 4)
    parser.add_argument('--mean', type=float, help='model name', default=-1)
    parser.add_argument('--std', type=float, help='model name', default=2)
    args = parser.parse_args()

    epoch = args.epoch
    gpus = []
    test_dir = args.test_dir
    result_dir = args.result_dir
    name = args.model_name
    min_size = args.min_size
    mean = args.mean
    std = args.std
    buildDir([result_dir])
    testDataset = DataLoader(TestDataset(test_dir, min_size=min_size), batch_size=1)

    G_model = Model(epoch, name=name, gpus=gpus).model.eval()

    # print(G_model)
    for root, dirs, files in os.walk(test_dir):
        imagesNames = files
        break

    test_input = Variable(requires_grad = False)
    if (len(gpus) > 0) & torch.cuda.is_available():
        test_input = Variable().cuda()

    for index, test_img in enumerate(testDataset):
        # batchsize, channel, row, col = test_img.size()
        # test_img = torch.nn.Upsample(scale_factor=1)(test_img)
        batchsize, channel, row, col = test_img.size()
        test_input.data.resize_(test_img.size()).copy_(test_img)  # test input data

        with torch.no_grad():
            y = G_model(test_input).cpu().reshape(1, row, col)

        y = transforms.Normalize([mean, mean, mean], [std, std, std])(y)  # (-1,1)=>(0,1)   mean = -1,std = 2

        y = transforms.ToPILImage()(y)

        y.save("%s/%s_Mask.png" % (result_dir, imagesNames[index][:-4]))
        print("%s/%s_Mask.png finished" % (result_dir, imagesNames[index][:-4]))
        # plt.imshow(y, cmap="gray")
        # plt.show()
