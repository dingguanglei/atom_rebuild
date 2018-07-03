# coding=utf-8
from mypackage.utils import Model
from tensorboardX import SummaryWriter
import torch
import torchvision
model = Model(50).model
writer = SummaryWriter(log_dir="log/network")

input = torch.autograd.Variable(torch.Tensor(1, 1, 256, 256), requires_grad=True)
writer.add_graph(model=model, input_to_model=(input,))

# model = torchvision.models.AlexNet(num_classes=10)
# # 准备写tensorboard, 必须放在'.to(device)'之前，不然会报错
# writer = SummaryWriter(log_dir="log/network")
# dummy_input = torch.autograd.Variable(torch.rand(1, 3, 227, 227))
# writer.add_graph(model=model, input_to_model=dummy_input)




