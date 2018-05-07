# Generative Adversarial Networks (GAN) example in PyTorch.
# See related blog post at https://medium.com/@devnag/generative-adversarial-networks-gans-in-50-lines-of-code-pytorch-e81b79659e3f#.sch4xgsa9
import numpy as np
import time
from math import log10
import torch
import torch.nn as nn
from torch.nn import Sequential, Conv2d, BatchNorm2d, LeakyReLU, ReLU, Dropout, Tanh, Upsample, init, Sigmoid, \
    ConvTranspose2d
from torch.autograd import Variable, grad
import torchvision.transforms as transforms
from torch.optim import Adam
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Dataset
import os
from skimage import img_as_float, io, data, exposure, img_as_float, img_as_ubyte, img_as_uint, color
from sklearn.model_selection import train_test_split
from tensorboardX import SummaryWriter
from scipy.ndimage import gaussian_filter
from skimage import filters
from skimage.morphology import reconstruction, dilation, disk, erosion
from skimage.exposure import exposure, equalize_hist
from network.network import *
from torch import cuda


# def save(model, filename='model/D.pkl'):
#     torch.save(model.state_dict(), filename)
#
#
# def load(model, filename='model/D.pkl'):
#     model.load_state_dict(torch.load(filename))
#
#
# class AtomDataset(Dataset):
#     """Face Landmarks dataset."""
#
#     def __init__(self, img_x_path="images/image/", img_y_path="images/mask/", num=128, transform=None, startPoint=0, ):
#
#         self.x_train, self.y_train = \
#             self.__readImages(img_x_path, img_y_path, startPoint=0, num=num)
#
#         self.transform = transform
#
#     def __readImages(self, imagePath, maskPath, startPoint=0, num=1000):
#
#         imageData = []
#         maskData = []
#         imagesNames = []
#         maskNames = []
#         for root, dirs, files in os.walk(imagePath):
#             imagesNames = files
#             break
#         for root, dirs, files in os.walk(maskPath):
#             maskNames = files
#             break
#
#         assert (startPoint < len(imagesNames) - 1)
#         if (startPoint + num - 1) > len(imagesNames) - 1:
#             num = len(imagesNames) - startPoint
#
#         for index in range(startPoint, startPoint + num):
#             imageData.append(data.imread(imagePath + imagesNames[index]) / 256)
#             maskData.append(data.imread(maskPath + maskNames[index]) / 256)
#         imageData = np.array(imageData, dtype=np.float32).reshape(-1, 256, 256, 1)
#         maskData = np.array(maskData, dtype=np.float32).reshape(-1, 256, 256, 1)
#         # x_train, x_test, y_train, y_test = train_test_split(imageData, maskData, test_size=test_size,
#         #                                                     random_state=33)
#         return imageData, maskData
#
#     def __len__(self):
#         return len(self.x_train)
#
#     def __getitem__(self, index):
#         x_img, y_img = self.x_train[index], self.y_train[index]
#         return x_img, y_img
#
#
# def prepareDataLoader(batch_size=32, num_workers=8, test_size=100, train_size=None):
#     imagesNames = []
#     maskNames = []
#     imagePath = "images/image/"
#     maskPath = "images/mask/"
#     for root, dirs, files in os.walk(imagePath):
#         imagesNames = files
#         break
#     for root, dirs, files in os.walk(maskPath):
#         maskNames = files
#         break
#     assert (len(imagesNames) == len(maskNames))
#     x_train, x_test, y_train, y_test = train_test_split(imagesNames, maskNames, shuffle=True, test_size=test_size,
#                                                         random_state=33, train_size=train_size)
#
#     class TrainDataset(Dataset):
#
#         def __init__(self, transform=None):
#             self.transform = transform
#             self.x_train = []
#             self.y_train = []
#             transform_list = [transforms.ToTensor(),  # (H x W x C)=> (C x H x W)
#                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
#             self.transform = transforms.Compose(transform_list)
#
#             for index in range(len(x_train)):
#                 self.x_train.append(data.imread(imagePath + x_train[index]) / 256)
#                 self.y_train.append(data.imread(maskPath + y_train[index]) / 256)
#             self.x_train = np.array(self.x_train, dtype=np.float32).reshape(-1, 256, 256, 1)
#             self.y_train = np.array(self.y_train, dtype=np.float32).reshape(-1, 256, 256, 1)
#
#         def __len__(self):
#             return len(self.x_train)
#
#         def __getitem__(self, index):
#             x_img_train, y_img_train = self.x_train[index], self.y_train[index]
#             x_img_train = self.transform(x_img_train)
#             y_img_train = self.transform(y_img_train)
#             return x_img_train, y_img_train
#
#     class TestDataset(Dataset):
#
#         def __init__(self, transform=None):
#             self.transform = transform
#             self.x_test = []
#             self.y_test = []
#             transform_list = [transforms.ToTensor(),  # (H x W x C)=> (C x H x W)
#                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
#             self.transform = transforms.Compose(transform_list)
#
#             for index in range(len(x_test)):
#                 self.x_test.append(data.imread(imagePath + x_test[index]) / 256)
#                 self.y_test.append(data.imread(maskPath + y_test[index]) / 256)
#             self.x_test = np.array(self.x_test, dtype=np.float32).reshape(-1, 256, 256, 1)
#             self.y_test = np.array(self.y_test, dtype=np.float32).reshape(-1, 256, 256, 1)
#
#         def __len__(self):
#             return len(self.x_test)
#
#         def __getitem__(self, index):
#             x_img_test, y_img_test = self.x_test[index], self.y_test[index]
#             x_img_test = self.transform(x_img_test)
#             y_img_test = self.transform(y_img_test)
#             return x_img_test, y_img_test
#
#     trainLoader = DataLoader(TrainDataset(), batch_size=batch_size, shuffle=True, num_workers=num_workers)
#     testLoader = DataLoader(TestDataset(), batch_size=batch_size, shuffle=True, num_workers=num_workers)
#     return trainLoader, testLoader
#
#
# class ElapsedTimer(object):
#     def __init__(self):
#         self.start_time = time.time()
#
#     def elapsed(self, sec):
#         if sec < 60:
#             return str(sec) + " sec"
#         elif sec < (60 * 60):
#             return str(sec / 60) + " min"
#         else:
#             return str(sec / (60 * 60)) + " hr"
#
#     def elapsed_time(self):
#         print("Elapsed: %s " % self.elapsed(time.time() - self.start_time))
#
#
# class GanTrain():
#     def __init__(self, gen, dis, loader):
#         self.gpu_avavailable = torch.cuda.is_available()
#
#         self.G = gen
#         self.D = dis
#
#         try:
#             self.G.load()
#             print("load D model... success!")
#         except:
#             print("load D model... fail!")
#
#         try:
#             self.D.load()
#             print("load G model... success!")
#         except:
#             print("load G model... fail!")
#
#         if self.gpu_avavailable:
#             self.G = gen.cuda()
#             self.D = dis.cuda()
#             if torch.cuda.device_count() > 1:
#                 print("Let's use", torch.cuda.device_count(), "GPUs!")
#                 # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
#                 self.G = torch.nn.DataParallel(gen)
#                 self.D = torch.nn.DataParallel(dis)
#         print(gen)
#         print(dis)
#         self.__weights_init(self.G)
#         self.__weights_init(self.D)
#         self.loader = loader
#         self.step = len(loader)  # 循环遍历一个epochs需要的steps数目
#         self.batch_size = loader.batch_size
#         self.shape = (256, 256, 1)
#         self.best_loss = 10000
#         self.D_losses = []
#         self.G_losses = []
#         ###########   LOSS & OPTIMIZER   ##########
#         lr = 1e-5
#         beta1 = (0.5, 0.9)
#         self.loss_BCE = nn.BCELoss()
#         self.loss_L1 = nn.L1Loss()
#         self.loss_MSE = nn.MSELoss()
#         self.optimizerD = torch.optim.RMSprop(self.D.parameters(), lr=lr)
#         self.optimizerG = torch.optim.RMSprop(self.G.parameters(), lr=lr)
#
#         # self.one = torch.FloatTensor([1])
#         # self.minus_one = torch.FloatTensor([1]) * -1
#         # if self.gpu_avavailable:
#         #     self.one = self.one.cuda()
#         #     self.minus_one = self.minus_one.cuda()
#
#     def __weights_init(self, m):
#         classname = m.__class__.__name__
#         if classname.find('Conv') != -1:
#             m.weight.data.normal_(0, 0.00001)
#         elif classname.find('BatchNorm2d') != -1 or classname.find('InstanceNorm2d') != -1:
#             m.weight.data.normal_(1.0, 0.02)
#             m.bias.data.fill_(0)
#
#     def gradient_penalty(self, netD, real_data, fake_data, LAMBDA=20):
#         # print real_data.size()
#         alpha = torch.rand(real_data.size()[0], real_data.size()[1], real_data.size()[2], real_data.size()[3])
#         # alpha = alpha.expand(real_data.size())
#         if self.gpu_avavailable:
#             alpha = alpha.cuda()
#
#         interpolates = alpha * real_data + ((1 - alpha) * fake_data)
#
#         if (torch.cuda.is_available()):
#             interpolates = interpolates.cuda()
#         interpolates = Variable(interpolates, requires_grad=True)
#
#         disc_interpolates = netD(interpolates)
#
#         gradients = grad(outputs=disc_interpolates, inputs=interpolates,
#                          grad_outputs=torch.ones(disc_interpolates.size()).cuda(
#                          ) if self.gpu_avavailable else torch.ones(
#                              disc_interpolates.size()),
#                          create_graph=True, retain_graph=True, only_inputs=True)[0]
#
#         gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * LAMBDA
#         return gradient_penalty
#
#     def train(self, epochs=20, save_interval=300, show=False):  # 循环遍历一个epochs需要的steps数目
#         writer = SummaryWriter(log_dir="log")
#         for j in range(epochs):
#             start_time = time.time()
#
#             self.__one_epoch_train(j, save_interval=save_interval, show=show, writer=writer)
#
#             pred_cost_time = (time.time() - start_time) * (epochs - j - 1)
#             timer = ElapsedTimer().elapsed(pred_cost_time)
#             print("All epochs Left time : %s" % timer)
#         writer.close()
#
#     def __one_epoch_train(self, current_epoch, save_interval=300, show=False, writer=None):
#         current_step = 0
#         dis_step = 20
#         total_step = current_epoch * self.step + current_step + 1
#         for x_train, y_train in self.loader:
#
#             start_time = time.time()
#             ############################
#             # (1) Update D network
#             ##########################
#             for p in self.D.parameters():  # reset requires_grad
#                 p.requires_grad = True  # they are set to False below in netG update
#
#             if (total_step < 50) or (total_step % 500):
#                 dis_step = 1
#
#             else:
#                 dis_step = 1
#
#             for d_index in range(dis_step):
#
#                 # for p in self.D.parameters():
#                 #     p.data.clamp_(-0.01, 0.01)
#
#                 if self.gpu_avavailable:
#                     input_data = Variable(x_train).cuda()
#                     real_data = Variable(y_train).cuda()
#                 else:
#                     input_data = Variable(x_train)
#                     real_data = Variable(y_train)
#
#                 with torch.no_grad():
#                     input_data = input_data
#                 fake_data = Variable(self.G(input_data))
#
#                 self.D.zero_grad()
#                 # train with real
#                 D_real = self.D(real_data, input_data)
#                 D_fake = self.D(fake_data, input_data)
#
#                 # train with gradient penalty
#                 # gradient_penalty = self.gradient_penalty(self.D, real_data, fake_data)
#                 # gradient_penalty.backward()
#
#                 self.D_loss = 0.5 * (torch.mean((D_real - 1) ** 2) + torch.mean(D_fake ** 2))
#                 self.D_losses.append(self.D_loss)
#                 # self.Wasserstein_D = D_real - D_fake
#                 self.D_loss.backward()
#                 self.optimizerD.step()
#                 writer.add_scalars("D_loss", {"train": self.D_loss}, len(self.D_losses) + 1)
#
#                 # print(self.D_loss)
#
#             ############################
#             # (2) Update G network
#             ###########################
#             for p in self.D.parameters():
#                 p.requires_grad = False  # to avoid computation
#
#             self.G.zero_grad()
#             input_data = Variable(x_train)
#             if self.gpu_avavailable:
#                 input_data = input_data.cuda()
#             fake_data = self.G(input_data)
#
#             G_fake = self.D(fake_data, input_data)
#             # G_fake.backward(self.minus_one)
#
#             # self.G_loss = - G_fake
#             self.G_loss = 0.5 * torch.mean((G_fake - 1) ** 2)
#             self.G_loss.backward()
#             self.optimizerG.step()
#             self.G_losses.append(self.G_loss)
#             writer.add_scalars("G_loss", {"train": self.G_loss}, len(self.G_losses) + 1)
#
#             log_mesg = "Epoch: %d ---- %d/%d:" % (current_epoch, current_step, self.step)
#             # log_mesg = "%s [Wasserstein_D: %f]" % (log_mesg, self.Wasserstein_D)
#             log_mesg = "%s [D loss: %f]" % (log_mesg, self.D_loss)
#             log_mesg = "%s  [G loss: %f]" % (log_mesg, self.G_loss)
#             print(log_mesg)
#             cost_time = time.time() - start_time
#             pred_cost_time = cost_time * (self.step - current_step - 1)
#             timer = ElapsedTimer().elapsed(pred_cost_time)
#
#             current_step = current_step + 1
#             print("Left time for 1 epoch: %s" % timer)
#
#             if (save_interval > 0):
#                 if (current_step + 1) % save_interval == 0:
#                     # 评估模型
#                     # # 保存和加载整个模型
#                     # torch.save(model_object, 'model.pkl')
#                     # model = torch.load('model.pkl')
#                     ## 仅保存和加载模型参数(推荐使用)
#                     # torch.save(model_object.state_dict(), 'params.pkl')
#                     # model_object.load_state_dict(torch.load('params.pkl'))
#
#                     if self.best_loss > self.G_loss or self.best_loss <= self.G_loss:
#                         # if False:
#                         save(self.D, 'model/D.pkl')
#                         save(self.G, 'model/G.pkl')
#
#                         # self.D.save()
#                         # self.G.save()
#                         print("savePoint:", current_step,
#                               "best loss is %f, new loss is %f. SAVE!" % (self.best_loss, self.G_loss))
#                         self.best_loss = self.G_loss
#
#                     print(current_step,
#                           "best loss is %f, new loss is %f. DO NOT SAVE!" % (self.best_loss, self.G_loss))
#                     loader = DataLoader(AtomDataset(startPoint=(0 + current_step) % 10000, num=1), batch_size=1)
#                     shape = (256, 256)
#                     for x, y in loader:
#                         x_for_G = x.permute(0, 3, 1, 2)
#                         x_for_G = Variable(x_for_G)
#                         if self.gpu_avavailable:
#                             x_for_G = x_for_G.cuda()
#                         self.G.eval()
#                         pred = self.G(x_for_G).cpu()
#                         pred = np.reshape(pred.detach().numpy(), shape)
#                         self.G.train()
#                         if show:
#                             plt.figure(figsize=(4, 10))
#                             plt.subplot(3, 1, 1)
#                             plt.imshow(np.reshape(x, shape), cmap='gray')
#                             plt.title("input")
#                             plt.axis('off')
#                             plt.tight_layout()
#                             plt.subplot(3, 1, 2)
#                             plt.imshow(pred, cmap='gray')
#                             plt.title("gen")
#                             plt.axis('off')
#                             plt.tight_layout()
#                             plt.subplot(3, 1, 3)
#                             plt.imshow(np.reshape(y, shape), cmap='gray')
#                             plt.title("y_test")
#                             plt.axis('off')
#                             plt.tight_layout()
#                             plt.show()
#                             plt.close('all')
#                         else:
#                             # fake = exposure.rescale_intensity(fake,in_range="image",out_range=(-1,1))
#                             x_filename = "train_plots/kgan_x_E%dS%d.png" % (current_epoch, current_step + 1)
#                             y_filename = "train_plots/kgan_y_E%dS%d.png" % (current_epoch, current_step + 1)
#                             g_filename = "train_plots/kgan_g_E%dS%d.png" % (current_epoch, current_step + 1)
#                             io.imsave(x_filename, np.reshape(x, shape))
#                             io.imsave(y_filename, np.reshape(y, shape))
#                             io.imsave(g_filename, pred)
#
#
# def train(training_data_loader, epoch, reala, realb):
#     lamb = 10
#     for iteration, batch in enumerate(training_data_loader, 1):
#         # forward
#         real_a_cpu, real_b_cpu = batch[0], batch[1]
#         real_a = reala.data.resize_(real_a_cpu.size()).copy_(real_a_cpu)
#         real_b = realb.data.resize_(real_b_cpu.size()).copy_(real_b_cpu)
#         fake_b = netG(real_a)
#
#         ############################
#         # (1) Update D network: maximize log(D(x,y)) + log(1 - D(x,G(x)))
#         ###########################
#
#         optimizerD.zero_grad()
#
#         # train with fake
#         fake_ab = torch.cat((real_a, fake_b), 1)
#         pred_fake = netD.forward(fake_ab.detach())
#         loss_d_fake = criterionGAN(pred_fake, False)
#
#         # train with real
#         real_ab = torch.cat((real_a, real_b), 1)
#         pred_real = netD.forward(real_ab)
#         loss_d_real = criterionGAN(pred_real, True)
#
#         # Combined loss
#         loss_d = (loss_d_fake + loss_d_real) * 0.5
#
#         loss_d.backward()
#
#         optimizerD.step()
#
#         ############################
#         # (2) Update G network: maximize log(D(x,G(x))) + L1(y,G(x))
#         ##########################
#         optimizerG.zero_grad()
#         # First, G(A) should fake the discriminator
#         fake_ab = torch.cat((real_a, fake_b), 1)
#         pred_fake = netD.forward(fake_ab)
#         loss_g_gan = criterionGAN(pred_fake, True)
#
#         # Second, G(A) = B
#         loss_g_l1 = criterionL1(fake_b, real_b) * lamb
#
#         loss_g = loss_g_gan + loss_g_l1
#
#         loss_g.backward()
#
#         optimizerG.step()
#
#         print("===> Epoch[{}]({}/{}): Loss_D: {:.4f} Loss_G: {:.4f}".format(
#             epoch, iteration, len(training_data_loader), loss_d.data[0], loss_g.data[0]))
#
#         steps = len(training_data_loader)
#         count = steps * (epoch - 1) + iteration
#         writer.add_scalars("Train Loss", {"Loss_D": loss_d.data[0], "Loss_G": loss_g.data[0]}, count)
#
#
# def test(testing_data_loader, netG, current_epoch=0):
#     avg_psnr = 0
#     len_test_data = len(testing_data_loader)
#     rolling_index = current_epoch % len_test_data
#     for input, real in testing_data_loader:
#         with torch.no_grad():
#             input = Variable(input)
#             real = Variable(real)
#         if cuda.is_available():
#             input = input.cuda()
#             real = real.cuda()
#         netG.eval()
#         prediction = netG(input)
#         netG.train()
#         mse = criterionMSE(prediction, real)
#         psnr = 10 * log10(1 / mse.data[0])
#         avg_psnr += psnr
#     shape = (256, 256)
#     avg_psnr = avg_psnr / len_test_data
#     print("===> Avg. PSNR: {:.4f} dB".format(avg_psnr))
#     writer.add_scalars("psnr", {"test": avg_psnr}, current_epoch)
#     input = input[rolling_index].cpu().detach().numpy().reshape(shape)
#     prediction = prediction[rolling_index].cpu().detach().numpy().reshape(shape)
#     real = real[rolling_index].cpu().detach().numpy().reshape(shape)
#     x_filename = "train_plots/ptgan_x_E%03d.png" % (current_epoch)
#     y_filename = "train_plots/ptgan_y_E%03d.png" % (current_epoch)
#     g_filename = "train_plots/ptgan_g_E%03d.png" % (current_epoch)
#     io.imsave(x_filename, input)
#     io.imsave(y_filename, real)
#     io.imsave(g_filename, prediction)
#
#
# def checkPoint(netG, netD, epoch):
#     if not os.path.exists("checkpoint"):
#         os.mkdir("checkpoint")
#     net_g_model_out_path = "checkpoint/netG_model_epoch_{}.pth".format(epoch)
#     net_d_model_out_path = "checkpoint/netD_model_epoch_{}.pth".format(epoch)
#     torch.save(netG.state_dict(), net_g_model_out_path)
#     torch.save(netD.state_dict(), net_d_model_out_path)
#     # torch.save(netG, net_g_model_out_path)
#     # torch.save(netD, net_d_model_out_path)
#     print("Checkpoint saved !")
#
#
# def loadCheckPoint(netG, netD, epoch):
#     net_g_model_out_path = "checkpoint/netG_model_epoch_{}.pth".format(epoch)
#     net_d_model_out_path = "checkpoint/netD_model_epoch_{}.pth".format(epoch)
#     netG.load_state_dict(torch.load(net_g_model_out_path))
#     netD.load_state_dict(torch.load(net_d_model_out_path))
#     print("Checkpoint loaded !")
#     return netG, netD
#
#
# writer = SummaryWriter(log_dir="log")
#
#
# class MyDataset(Dataset):
#
#     def __init__(self, imagePath, maskPath=None):
#         self.transform = None
#         self.x = []
#         self.y = []
#         self.imagePath = imagePath
#         self.maskPath = maskPath
#         self.imagesNames = []
#         self.maskNames = []
#         transform_list = [transforms.ToTensor(),  # (H x W x C)=> (C x H x W)
#                           transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))] # for each channel
#         self.transform = transforms.Compose(transform_list)
#
#         for root, dirs, files in os.walk(imagePath):
#             self.imagesNames = files
#             break
#         for index in range(len(self.imagesNames)):
#             self.x.append(data.imread(imagePath + self.imagesNames[index], as_grey=True))
#         self.x = np.array(self.x, dtype=np.float32).reshape(self.x)
#
#         if maskPath is not None:
#             for root, dirs, files in os.walk(maskPath):
#                 self.maskNames = files
#                 break
#             for index in range(len(self.maskNames)):
#                 self.y.append(data.imread(maskPath + self.maskNames[index], as_grey=True))
#             self.y = np.array(self.y, dtype=np.float32)
#
#     def __len__(self):
#         return len(self.x)
#
#     def __getitem__(self, index):
#         x_img = self.x[index]
#         x_img = self.transform(x_img)
#         x_name = self.imagesNames[index]
#         if self.maskPath is not None:
#             y_img = self.y[index]
#             y_img = self.transform(y_img)
#             y_name = self.maskNames[index]
#             return x_img, x_name, y_img, y_name
#             return x_img, x_name


if __name__ == '__main__':
    # dataset = AtomDataset()
    # dataset：Dataset类型，从其中加载数据
    # batch_size：int，可选。每个batch加载多少样本
    # shuffle：bool，可选。为True时表示每个epoch都对数据进行洗牌
    # sampler：Sampler，可选。从数据集中采样样本的方法。
    # num_workers：int，可选。加载数据时使用多少子进程。默认值为0，表示在主进程中加载数据。
    # collate_fn：callable，可选。
    # pin_memory：bool，可选
    # drop_last：bool，可选。True表示如果最后剩下不完全的batch, 丢弃。False表示不丢弃。
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
    input_nc = 1
    output_nc = 1

    ngf = ndf = 32
    lr = 1e-5
    beta1 = 0.9
    batchSize = 64
    nepochs = 500
    trainLoader = None
    testLoader = None
    GPUS = [0]
    # trainLoader, testLoader = prepareDataLoader(batch_size=batchSize, test_size=100, train_size=None)
    testLoader = DataLoader(MyDataset("process_plots/O/"), batch_size=1, shuffle=False, num_workers=2)
    for index, test_img in enumerate(testLoader):
        img = test_img[0]
        name = test_img[1][0]
        if torch.cuda.is_available():
            img = img.cuda()
        result_img = netG(img)
        filename = "process_results/%s.png" % (name)
        result_img = result_img.cpu().detach().numpy().reshape(256, 256)
        # plt.imshow(result_img, cmap="gray")
        # plt.show()
        io.imsave(filename, result_img)
    exit(1)
    print('===> Building model')
    netG = define_G(input_nc, output_nc, ngf, 'batch', False, GPUS)
    netD = define_D(input_nc + output_nc, ndf, 'batch', False, GPUS)
    netG, netD = loadCheckPoint(netG, netD, 300)

    criterionGAN = GANLoss()
    criterionL1 = nn.L1Loss()
    criterionMSE = nn.MSELoss()

    # setup optimizer
    optimizerG = Adam(netG.parameters(), lr=lr, betas=(beta1, 0.999))
    optimizerD = Adam(netD.parameters(), lr=lr, betas=(beta1, 0.999))

    print('---------- Networks initialized -------------')
    print_network(netG)
    print_network(netD)
    print('-----------------------------------------------')

    real_a = Variable()
    real_b = Variable()

    if cuda.is_available():
        netD = netD.cuda()
        netG = netG.cuda()
        criterionGAN = criterionGAN.cuda()
        criterionL1 = criterionL1.cuda()
        criterionMSE = criterionMSE.cuda()
        real_a = real_a.cuda()
        real_b = real_b.cuda()

    # real_a = Variable(real_a)
    # real_b = Variable(real_b)
    startEpoch = 100
    # netG, netD = loadCheckPoint(netG, netD, startEpoch)

    for epoch in range(startEpoch, nepochs + 1):
        train(trainLoader, epoch, real_a, real_b)
        test(testLoader, netG, current_epoch=epoch)
        if epoch % 100 == 0:
            checkPoint(netG, netD, epoch)
