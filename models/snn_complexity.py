from spikingjelly.activation_based import neuron, layer, functional
import torch.nn as nn
import torch.nn.functional as F
import sys
sys.path.append('.')

import numpy as np
import random
import torch
from torch import nn
import torch.nn.functional as F
from utils.options import args_parser
import copy
from thop import profile





class Surrogate_BP_Function(torch.autograd.Function):


    @staticmethod
    def forward(ctx, input):   # mem  or  mem * threshold
        ctx.save_for_backward(input)
        out = torch.zeros_like(input).cuda()
        out[input > 0] = 1.0
        return out

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad = grad_input * 0.3 * F.threshold(1.0 - torch.abs(input), 0, 0)   # mem  or  mem * threshold
        return grad


def PoissonGen(inp, rescale_fac=2.0):
    rand_inp = torch.rand_like(inp).cuda()
    return torch.mul(torch.le(rand_inp * rescale_fac, torch.abs(inp)).float(), torch.sign(inp))



#实验只用最小模型的效果
class SmallSNNCifar(nn.Module):
    def __init__(self, args):
        super(SmallSNNCifar, self).__init__()
        self.T = args.T
        self.th = args.threshold
        self.th2 = args.threshold2
        self.th3 = args.threshold3

        self.conv1 = layer.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = layer.BatchNorm2d(32)
        self.sn1 = neuron.IFNode(v_threshold=self.th)
        self.conv2 = layer.Conv2d(32, 32, kernel_size=3, padding=1)
        self.bn2 = layer.BatchNorm2d(32)
        self.sn2 = neuron.IFNode(v_threshold=self.th)
        self.pool = layer.MaxPool2d(kernel_size=2)

        self.flat = layer.Flatten()

        self.fc1 = layer.Linear(32 * 16 * 16, 128)
        self.sn3 = neuron.IFNode(v_threshold=self.th)
        self.fc2 = layer.Linear(128, args.num_classes)
        self.sn4 = neuron.IFNode(v_threshold=self.th)

        # 初始化权重
        self._initialize_weights()

        # 多步模式
        functional.set_step_mode(self, step_mode='m')

    # 初始化权重
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.Conv2d)):
                torch.nn.init.kaiming_normal_(m.weight.data)
                if m.bias is not None:
                    torch.nn.init.zeros_(m.bias.data)
            elif isinstance(m, nn.Linear):
                torch.nn.init.kaiming_normal_(m.weight.data)
                if m.bias is not None:
                    torch.nn.init.zeros_(m.bias.data)
            elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
                m.weight.data.fill_(1)
                torch.nn.init.zeros_(m.bias.data)



    def forward(self, x):
        # 添加时间步
        x = x.unsqueeze(0).repeat(self.T, 1, 1, 1, 1)

        x = self.sn1(self.bn1(self.conv1(x)))
        x = self.sn2(self.bn2(self.conv2(x)))
        x = self.pool(x)


        # x = x.view(-1, 32 * 16 * 16)
        x = self.flat(x)
        x = self.sn3(self.fc1(x))
        x = self.sn4(self.fc2(x))

        x = x.mean(0)
        return x

    def spiking_encoder(self, x):
        return self.bn1(self.sn1(self.conv1(x)))


class SmallSNNFmnist(nn.Module):
    def __init__(self, args):
        super(SmallSNNFmnist, self).__init__()
        self.T = args.T
        self.th = args.threshold
        self.th2 = args.threshold2
        self.th3 = args.threshold3

        self.conv1 = layer.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1 = layer.BatchNorm2d(32)
        self.sn1 = neuron.IFNode(v_threshold=self.th)
        self.conv2 = layer.Conv2d(32, 32, kernel_size=3, padding=1)
        self.bn2 = layer.BatchNorm2d(32)
        self.sn2 = neuron.IFNode(v_threshold=self.th)
        self.pool = layer.MaxPool2d(kernel_size=2)

        self.flat = layer.Flatten()

        self.fc1 = layer.Linear(32 * 14 * 14, 128)
        self.sn3 = neuron.IFNode(v_threshold=self.th)
        self.fc = layer.Linear(128, args.num_classes)
        self.sn4 = neuron.IFNode(v_threshold=self.th)

        # 初始化权重
        self._initialize_weights()

        # 多步模式
        functional.set_step_mode(self, step_mode='m')

    # 初始化权重
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.Conv2d)):
                torch.nn.init.kaiming_normal_(m.weight.data)
                if m.bias is not None:
                    torch.nn.init.zeros_(m.bias.data)
            elif isinstance(m, nn.Linear):
                torch.nn.init.kaiming_normal_(m.weight.data)
                if m.bias is not None:
                    torch.nn.init.zeros_(m.bias.data)
            elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
                m.weight.data.fill_(1)
                torch.nn.init.zeros_(m.bias.data)



    def forward(self, x):
        # 添加时间步
        x = x.unsqueeze(0).repeat(self.T, 1, 1, 1, 1)

        x = self.sn1(self.bn1(self.conv1(x)))
        x = self.sn2(self.bn2(self.conv2(x)))
        x = self.pool(x)


        # x = x.view(-1, 32 * 16 * 16)
        x = self.flat(x)
        x = self.sn3(self.fc1(x))
        x = self.sn4(self.fc(x))

        x = x.mean(0)
        return x

    def spiking_encoder(self, x):
        return self.bn1(self.sn1(self.conv1(x)))



# 7层卷积模型
class SNN_VGG9(nn.Module):
    def __init__(self, args):
        super(SNN_VGG9, self).__init__()
        self.T = args.T
        self.th = args.threshold
        self.th2 = args.threshold2
        self.th3 = args.threshold3
        self.tau = args.tau

        self.conv1 = layer.Conv2d(3, 64, kernel_size=3, padding=1)
        self.bn1 = layer.BatchNorm2d(64)
        self.sn1 = neuron.LIFNode(v_threshold=self.th, tau=self.tau)
        self.conv2 = layer.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn2 = layer.BatchNorm2d(64)
        self.sn2 = neuron.LIFNode(v_threshold=self.th, tau=self.tau)
        self.pool1 = layer.MaxPool2d(kernel_size=2)

        self.conv3 = layer.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = layer.BatchNorm2d(128)
        self.sn3 = neuron.LIFNode(v_threshold=self.th, tau=self.tau)
        self.conv4 = layer.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn4 = layer.BatchNorm2d(128)
        self.sn4 = neuron.LIFNode(v_threshold=self.th, tau=self.tau)
        self.pool2 = layer.MaxPool2d(kernel_size=2)

        self.conv5 = layer.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn5 = layer.BatchNorm2d(256)
        self.sn5 = neuron.LIFNode(v_threshold=self.th, tau=self.tau)
        self.conv6 = layer.Conv2d(256, 256, kernel_size=3, padding=1)
        self.bn6 = layer.BatchNorm2d(256)
        self.sn6 = neuron.LIFNode(v_threshold=self.th, tau=self.tau)


        self.conv7 = layer.Conv2d(256, 256, kernel_size=3, padding=1)
        self.bn7 = layer.BatchNorm2d(256)
        self.sn7 = neuron.LIFNode(v_threshold=self.th, tau=self.tau)
        self.pool3 = layer.MaxPool2d(kernel_size=2)

        self.flat = layer.Flatten()
        self.global_fc1 = layer.Linear(256 * 4 * 4, 1024)
        # self.bn7 = nn.BatchNorm2d(128)
        self.sn8 = neuron.LIFNode(v_threshold=1., tau=self.tau)
        self.global_fc2 = layer.Linear(1024, args.num_classes)
        # self.bn8 = nn.BatchNorm2d(args.num_classes)
        self.sn9 = neuron.LIFNode(v_threshold=1., tau=self.tau)

        # 初始化权重
        self._initialize_weights()

        # 多步模式
        functional.set_step_mode(self, step_mode='m')

    # 初始化权重
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.Conv2d)):
                torch.nn.init.kaiming_normal_(m.weight.data)
                if m.bias is not None:
                    torch.nn.init.zeros_(m.bias.data)
            elif isinstance(m, nn.Linear):
                torch.nn.init.kaiming_normal_(m.weight.data)
                if m.bias is not None:
                    torch.nn.init.zeros_(m.bias.data)
            elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
                m.weight.data.fill_(1)
                torch.nn.init.zeros_(m.bias.data)



    def forward(self, x):

        # x.shape = [N, C, H, W]
        x = x.unsqueeze(0).repeat(self.T, 1, 1, 1, 1)

        x = self.sn1(self.bn1(self.conv1(x)))
        x = self.sn2(self.bn2(self.conv2(x)))
        x = self.pool1(x)

        x = self.sn3(self.bn3(self.conv3(x)))
        x = self.sn4(self.bn4(self.conv4(x)))
        x = self.pool2(x)

        # 更改sn5sn6查看数据
        # x = self.sn5(self.bn5(self.conv5(x)))
        # x = self.sn6(self.bn6(self.conv6(x)))
        x = self.conv5(x)
        x = self.bn5(x)
        x = self.sn5(x)
        x = self.conv6(x)
        x = self.bn6(x)
        x = self.sn6(x)
        # x = self.flat(self.pool3(x))
        x = self.conv7(x)
        x = self.bn7(x)
        x = self.sn7(x)
        x = self.pool3(x)
        x = self.flat(x)


        # x = x.mean(0)

        # x = x.view(-1, 128 * 4 * 4)
        # x = x.view(x.size(0), -1)
        # 经过sn8后数据全无 更改代码查看数据
        # x = self.sn7(self.global_fc1(x))
        # x = self.sn8(self.global_fc2(x))
        x = self.global_fc1(x)
        # x = self.bn7(x)
        x = self.sn8(x)
        x = self.global_fc2(x)
        # x = self.bn8(x)
        x = self.sn9(x)

        x = x.mean(0)

        return x



#完整模型
class EnhancedSNNCifar(nn.Module):
    def __init__(self, args):
        super(EnhancedSNNCifar, self).__init__()
        self.T = args.T
        self.th = args.threshold
        self.th2 = args.threshold2
        self.th3 = args.threshold3
        self.tau = args.tau

        self.conv1 = layer.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = layer.BatchNorm2d(32)
        self.sn1 = neuron.LIFNode(v_threshold=self.th, tau=self.tau)
        self.conv2 = layer.Conv2d(32, 32, kernel_size=3, padding=1)
        self.bn2 = layer.BatchNorm2d(32)
        self.sn2 = neuron.LIFNode(v_threshold=self.th, tau=self.tau)
        self.pool1 = layer.MaxPool2d(kernel_size=2)

        self.conv3 = layer.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn3 = layer.BatchNorm2d(64)
        self.sn3 = neuron.LIFNode(v_threshold=self.th, tau=self.tau)
        self.conv4 = layer.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn4 = layer.BatchNorm2d(64)
        self.sn4 = neuron.LIFNode(v_threshold=self.th, tau=self.tau)
        self.pool2 = layer.MaxPool2d(kernel_size=2)

        self.conv5 = layer.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn5 = layer.BatchNorm2d(128)
        self.sn5 = neuron.LIFNode(v_threshold=self.th, tau=self.tau)
        self.conv6 = layer.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn6 = layer.BatchNorm2d(128)
        self.sn6 = neuron.LIFNode(v_threshold=self.th, tau=self.tau)
        self.pool3 = layer.MaxPool2d(kernel_size=2)

        self.flat = layer.Flatten()
        self.global_fc1 = layer.Linear(128 * 4 * 4, 128)
        # self.bn7 = nn.BatchNorm2d(128)
        self.sn7 = neuron.LIFNode(v_threshold=1., tau=self.tau)
        self.global_fc2 = layer.Linear(128, args.num_classes)
        # self.bn8 = nn.BatchNorm2d(args.num_classes)
        self.sn8 = neuron.LIFNode(v_threshold=1., tau=self.tau)

        # 初始化权重
        self._initialize_weights()

        # 多步模式
        functional.set_step_mode(self, step_mode='m')

    # 初始化权重
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.Conv2d)):
                torch.nn.init.kaiming_normal_(m.weight.data)
                if m.bias is not None:
                    torch.nn.init.zeros_(m.bias.data)
            elif isinstance(m, nn.Linear):
                torch.nn.init.kaiming_normal_(m.weight.data)
                if m.bias is not None:
                    torch.nn.init.zeros_(m.bias.data)
            elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
                m.weight.data.fill_(1)
                torch.nn.init.zeros_(m.bias.data)



    def forward(self, x):

        # x.shape = [N, C, H, W]
        x = x.unsqueeze(0).repeat(self.T, 1, 1, 1, 1)

        x = self.sn1(self.bn1(self.conv1(x)))
        x = self.sn2(self.bn2(self.conv2(x)))
        x = self.pool1(x)

        x = self.sn3(self.bn3(self.conv3(x)))
        x = self.sn4(self.bn4(self.conv4(x)))
        x = self.pool2(x)

        # 更改sn5sn6查看数据
        # x = self.sn5(self.bn5(self.conv5(x)))
        # x = self.sn6(self.bn6(self.conv6(x)))
        x = self.conv5(x)
        x = self.bn5(x)
        x = self.sn5(x)
        x = self.conv6(x)
        x = self.bn6(x)
        x = self.sn6(x)
        # x = self.flat(self.pool3(x))
        x = self.pool3(x)
        x = self.flat(x)


        # x = x.mean(0)

        # x = x.view(-1, 128 * 4 * 4)
        # x = x.view(x.size(0), -1)
        # 经过sn8后数据全无 更改代码查看数据
        # x = self.sn7(self.global_fc1(x))
        # x = self.sn8(self.global_fc2(x))
        x = self.global_fc1(x)
        # x = self.bn7(x)
        x = self.sn7(x)
        x = self.global_fc2(x)
        # x = self.bn8(x)
        x = self.sn8(x)

        x = x.mean(0)

        return x

    def spiking_encoder(self, x):
        return self.bn1(self.sn1(self.conv1(x)))


class EnhancedSNNFmnist(nn.Module):
    def __init__(self, args):
        super(EnhancedSNNFmnist, self).__init__()
        self.T = args.T
        self.th = args.threshold
        self.th2 = args.threshold2
        self.th3 = args.threshold3

        self.conv1 = layer.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1 = layer.BatchNorm2d(32)
        self.sn1 = neuron.IFNode(v_threshold=self.th)
        self.conv2 = layer.Conv2d(32, 32, kernel_size=3, padding=1)
        self.bn2 = layer.BatchNorm2d(32)
        self.sn2 = neuron.IFNode(v_threshold=self.th)
        self.pool1 = layer.MaxPool2d(kernel_size=2)

        self.conv3 = layer.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn3 = layer.BatchNorm2d(64)
        self.sn3 = neuron.IFNode(v_threshold=self.th)
        self.conv4 = layer.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn4 = layer.BatchNorm2d(64)
        self.sn4 = neuron.IFNode(v_threshold=self.th)
        self.pool2 = layer.MaxPool2d(kernel_size=2)

        self.conv5 = layer.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn5 = layer.BatchNorm2d(128)
        self.sn5 = neuron.IFNode(v_threshold=self.th)
        self.conv6 = layer.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn6 = layer.BatchNorm2d(128)
        self.sn6 = neuron.IFNode(v_threshold=self.th)
        # self.pool3 = layer.MaxPool2d(kernel_size=2)

        self.flat = layer.Flatten()
        self.global_fc1 = layer.Linear(128 * 7 * 7, 128)
        # self.bn7 = nn.BatchNorm2d(128)
        self.sn7 = neuron.IFNode(v_threshold=1.)
        self.global_fc2 = layer.Linear(128, args.num_classes)
        # self.bn8 = nn.BatchNorm2d(args.num_classes)
        self.sn8 = neuron.IFNode(v_threshold=1.)

        # 初始化权重
        self._initialize_weights()

        # 多步模式
        functional.set_step_mode(self, step_mode='m')

    # 初始化权重
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.Conv2d)):
                torch.nn.init.kaiming_normal_(m.weight.data)
                if m.bias is not None:
                    torch.nn.init.zeros_(m.bias.data)
            elif isinstance(m, nn.Linear):
                torch.nn.init.kaiming_normal_(m.weight.data)
                if m.bias is not None:
                    torch.nn.init.zeros_(m.bias.data)
            elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
                m.weight.data.fill_(1)
                torch.nn.init.zeros_(m.bias.data)



    def forward(self, x):

        # x.shape = [N, C, H, W]
        x = x.unsqueeze(0).repeat(self.T, 1, 1, 1, 1)

        x = self.sn1(self.bn1(self.conv1(x)))
        x = self.sn2(self.bn2(self.conv2(x)))
        x = self.pool1(x)

        x = self.sn3(self.bn3(self.conv3(x)))
        x = self.sn4(self.bn4(self.conv4(x)))
        x = self.pool2(x)

        # 更改sn5sn6查看数据
        # x = self.sn5(self.bn5(self.conv5(x)))
        # x = self.sn6(self.bn6(self.conv6(x)))
        x = self.conv5(x)
        x = self.bn5(x)
        x = self.sn5(x)
        x = self.conv6(x)
        x = self.bn6(x)
        x = self.sn6(x)
        # x = self.flat(self.pool3(x))
        # x = self.pool3(x)
        x = self.flat(x)


        # x = x.mean(0)

        # x = x.view(-1, 128 * 4 * 4)
        # x = x.view(x.size(0), -1)
        # 经过sn8后数据全无 更改代码查看数据
        # x = self.sn7(self.global_fc1(x))
        # x = self.sn8(self.global_fc2(x))
        x = self.global_fc1(x)
        # x = self.bn7(x)
        x = self.sn7(x)
        x = self.global_fc2(x)
        # x = self.bn8(x)
        x = self.sn8(x)

        x = x.mean(0)

        return x

#分层模型
class SNNCifarComplexity(nn.Module):
    def __init__(self, args, complexity):
        super(SNNCifarComplexity, self).__init__()
        self.T = args.T
        self.th = args.threshold
        self.th2 = args.threshold2
        self.th3 = args.threshold3
        self.cplx = complexity
        self.tau = args.tau


        self.conv1 = layer.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = layer.BatchNorm2d(32)
        self.sn1 = neuron.LIFNode(v_threshold=self.th, tau=self.tau)
        self.conv2 = layer.Conv2d(32, 32, kernel_size=3, padding=1)
        self.bn2 = layer.BatchNorm2d(32)
        self.sn2 = neuron.LIFNode(v_threshold=self.th, tau=self.tau)

        self.pool = layer.MaxPool2d(kernel_size=2)
        # self.sn3 = neuron.IFNode()


        if self.cplx == 1:
            self.flat = layer.Flatten()

            self.fc1 = layer.Linear(32 * 16 * 16, 128)
            self.sn7 = neuron.LIFNode(v_threshold=self.th, tau=self.tau)
            self.fc = layer.Linear(128, args.num_classes)
            self.sn10 = neuron.LIFNode(v_threshold=self.th, tau=self.tau)

        elif self.cplx == 2:
            self.conv3 = layer.Conv2d(32, 64, kernel_size=3, padding=1)
            self.bn3 = layer.BatchNorm2d(64)
            self.sn3 = neuron.LIFNode(v_threshold=self.th, tau=self.tau)

            self.flat = layer.Flatten()
            self.fc2 = layer.Linear(64 * 8 * 8, 128)
            self.sn8 = neuron.LIFNode(v_threshold=self.th, tau=self.tau)
            self.fc = layer.Linear(128, args.num_classes)
            self.sn10 = neuron.LIFNode(v_threshold=self.th, tau=self.tau)


        elif self.cplx == 3:
            self.conv3 = layer.Conv2d(32, 64, kernel_size=3, padding=1)
            self.bn3 = layer.BatchNorm2d(64)
            self.sn3 = neuron.LIFNode(v_threshold=self.th, tau=self.tau)
            self.conv4 = layer.Conv2d(64, 64, kernel_size=3, padding=1)
            self.bn4 = layer.BatchNorm2d(64)
            self.sn4 = neuron.LIFNode(v_threshold=self.th2, tau=self.tau)


            self.flat = layer.Flatten()
            self.fc2 = layer.Linear(64 * 8 * 8, 128)
            self.sn8 = neuron.LIFNode(v_threshold=self.th2, tau=self.tau)
            self.fc = layer.Linear(128, args.num_classes)
            self.sn10 = neuron.LIFNode(v_threshold=self.th2, tau=self.tau)


        elif self.cplx == 4:
            self.conv3 = layer.Conv2d(32, 64, kernel_size=3, padding=1)
            self.bn3 = layer.BatchNorm2d(64)
            self.sn3 = neuron.LIFNode(v_threshold=self.th, tau=self.tau)
            self.conv4 = layer.Conv2d(64, 64, kernel_size=3, padding=1)
            self.bn4 = layer.BatchNorm2d(64)
            self.sn4 = neuron.LIFNode(v_threshold=self.th2, tau=self.tau)


            self.conv5 = layer.Conv2d(64, 128, kernel_size=3, padding=1)
            self.bn5 = layer.BatchNorm2d(128)
            self.sn5 = neuron.LIFNode(v_threshold=self.th3, tau=self.tau)

            self.flat = layer.Flatten()
            self.fc3 = layer.Linear(128 * 4 * 4, 128)
            self.sn9 = neuron.LIFNode(v_threshold=self.th3, tau=self.tau)
            self.fc = layer.Linear(128, args.num_classes)
            self.sn10 = neuron.LIFNode(v_threshold=self.th3, tau=self.tau)

        elif self.cplx == 5:
            self.conv3 = layer.Conv2d(32, 64, kernel_size=3, padding=1)
            self.bn3 = layer.BatchNorm2d(64)
            self.sn3 = neuron.LIFNode(v_threshold=self.th, tau=self.tau)
            self.conv4 = layer.Conv2d(64, 64, kernel_size=3, padding=1)
            self.bn4 = layer.BatchNorm2d(64)
            self.sn4 = neuron.LIFNode(v_threshold=self.th2, tau=self.tau)


            self.conv5 = layer.Conv2d(64, 128, kernel_size=3, padding=1)
            self.bn5 = layer.BatchNorm2d(128)
            self.sn5 = neuron.LIFNode(v_threshold=self.th3, tau=self.tau)
            self.conv6 = layer.Conv2d(128, 128, kernel_size=3, padding=1)
            self.bn6 = layer.BatchNorm2d(128)
            self.sn6 = neuron.LIFNode(v_threshold=self.th3, tau=self.tau)


            self.flat = layer.Flatten()
            self.global_fc1 = layer.Linear(128 * 4 * 4, 128)
            self.sn11 = neuron.LIFNode(v_threshold=self.th3, tau=self.tau)
            self.global_fc2 = layer.Linear(128, args.num_classes)
            self.sn12 = neuron.LIFNode(v_threshold=self.th3, tau=self.tau)


        # 多步模式
        functional.set_step_mode(self, step_mode='m')

        # 初始化权重
        self._initialize_weights()

    # 初始化权重
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.Conv2d)):
                torch.nn.init.kaiming_normal_(m.weight.data)
                if m.bias is not None:
                    torch.nn.init.zeros_(m.bias.data)
            elif isinstance(m, nn.Linear):
                torch.nn.init.kaiming_normal_(m.weight.data)
                if m.bias is not None:
                    torch.nn.init.zeros_(m.bias.data)
            elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
                m.weight.data.fill_(1)
                torch.nn.init.zeros_(m.bias.data)




    def forward(self, x):
        # 添加时间步
        x = x.unsqueeze(0).repeat(self.T, 1, 1, 1, 1)

        x = self.sn1(self.bn1(self.conv1(x)))
        x = self.sn2(self.bn2(self.conv2(x)))
        x = self.pool(x)

        if self.cplx == 1:

            # x = x.view(-1, 32 * 16 * 16)
            x = self.flat(x)
            x = self.sn7(self.fc1(x))
            x = self.sn10(self.fc(x))

            x = x.mean(0)
            return x

        x = self.sn3(self.bn3(self.conv3(x)))

        if self.cplx == 2:
            x = self.pool(x)
            # x = x.view(-1, 64 * 8 * 8)
            x = self.flat(x)
            x = self.sn8(self.fc2(x))
            x = self.sn10(self.fc(x))

            x = x.mean(0)
            return x

        x = self.sn4(self.bn4(self.conv4(x)))
        x = self.pool(x)

        if self.cplx == 3:
            # x = x.view(-1, 64 * 8 * 8)
            x = self.flat(x)
            x = self.sn8(self.fc2(x))
            x = self.sn10(self.fc(x))

            x = x.mean(0)
            return x

        x = self.sn5(self.bn5(self.conv5(x)))

        if self.cplx == 4:
            x = self.pool(x)
            # x = x.view(-1, 128 * 4 * 4)
            x = self.flat(x)
            x = self.sn9(self.fc3(x))
            x = self.sn10(self.fc(x))

            x = x.mean(0)
            return x

        x = self.sn6(self.bn6(self.conv6(x)))
        x = self.pool(x)

        if self.cplx == 5:
            # x = x.view(-1, 128 * 4 * 4)
            x = self.flat(x)
            x = self.sn11(self.global_fc1(x))
            x = self.sn12(self.global_fc2(x))

            x = x.mean(0)
            return x


class SNNVGG9Complexity(nn.Module):
    def __init__(self, args, complexity):
        super(SNNVGG9Complexity, self).__init__()
        self.T = args.T
        self.th = args.threshold
        self.th2 = args.threshold2
        self.th3 = args.threshold3
        self.cplx = complexity
        self.tau = args.tau


        self.conv1 = layer.Conv2d(3, 64, kernel_size=3, padding=1)
        self.bn1 = layer.BatchNorm2d(64)
        self.sn1 = neuron.LIFNode(v_threshold=self.th, tau=self.tau)
        self.conv2 = layer.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn2 = layer.BatchNorm2d(64)
        self.sn2 = neuron.LIFNode(v_threshold=self.th, tau=self.tau)
        self.conv3 = layer.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = layer.BatchNorm2d(128)
        self.sn3 = neuron.LIFNode(v_threshold=self.th, tau=self.tau)


        self.pool = layer.MaxPool2d(kernel_size=2)
        # self.sn3 = neuron.IFNode()


        if self.cplx == 1:
            self.flat = layer.Flatten()

            self.fc1 = layer.Linear(128 * 16 * 16, 128)
            self.sn8 = neuron.LIFNode(v_threshold=self.th, tau=self.tau)
            self.fc = layer.Linear(128, args.num_classes)
            self.sn11 = neuron.LIFNode(v_threshold=self.th, tau=self.tau)

        elif self.cplx == 2:
            self.conv4 = layer.Conv2d(128, 128, kernel_size=3, padding=1)
            self.bn4 = layer.BatchNorm2d(128)
            self.sn4 = neuron.LIFNode(v_threshold=self.th, tau=self.tau)

            self.flat = layer.Flatten()
            self.fc2 = layer.Linear(128 * 8 * 8, 128)
            self.sn9 = neuron.LIFNode(v_threshold=self.th, tau=self.tau)
            self.fc = layer.Linear(128, args.num_classes)
            self.sn11 = neuron.LIFNode(v_threshold=self.th, tau=self.tau)


        elif self.cplx == 3:
            self.conv4 = layer.Conv2d(128, 128, kernel_size=3, padding=1)
            self.bn4 = layer.BatchNorm2d(128)
            self.sn4 = neuron.LIFNode(v_threshold=self.th, tau=self.tau)
            self.conv5 = layer.Conv2d(128, 256, kernel_size=3, padding=1)
            self.bn5 = layer.BatchNorm2d(256)
            self.sn5 = neuron.LIFNode(v_threshold=self.th, tau=self.tau)


            self.flat = layer.Flatten()
            self.fc2 = layer.Linear(256 * 8 * 8, 128)
            self.sn9 = neuron.LIFNode(v_threshold=self.th2, tau=self.tau)
            self.fc = layer.Linear(128, args.num_classes)
            self.sn11 = neuron.LIFNode(v_threshold=self.th2, tau=self.tau)


        elif self.cplx == 4:
            self.conv4 = layer.Conv2d(128, 128, kernel_size=3, padding=1)
            self.bn4 = layer.BatchNorm2d(128)
            self.sn4 = neuron.LIFNode(v_threshold=self.th, tau=self.tau)
            self.conv5 = layer.Conv2d(128, 256, kernel_size=3, padding=1)
            self.bn5 = layer.BatchNorm2d(256)
            self.sn5 = neuron.LIFNode(v_threshold=self.th, tau=self.tau)

            self.conv6 = layer.Conv2d(256, 256, kernel_size=3, padding=1)
            self.bn6 = layer.BatchNorm2d(256)
            self.sn6 = neuron.LIFNode(v_threshold=self.th, tau=self.tau)

            self.flat = layer.Flatten()
            self.fc3 = layer.Linear(256 * 4 * 4, 128)
            self.sn10 = neuron.LIFNode(v_threshold=self.th3, tau=self.tau)
            self.fc = layer.Linear(128, args.num_classes)
            self.sn11 = neuron.LIFNode(v_threshold=self.th3, tau=self.tau)

        elif self.cplx == 5:
            self.conv4 = layer.Conv2d(128, 128, kernel_size=3, padding=1)
            self.bn4 = layer.BatchNorm2d(128)
            self.sn4 = neuron.LIFNode(v_threshold=self.th, tau=self.tau)
            self.conv5 = layer.Conv2d(128, 256, kernel_size=3, padding=1)
            self.bn5 = layer.BatchNorm2d(256)
            self.sn5 = neuron.LIFNode(v_threshold=self.th, tau=self.tau)

            self.conv6 = layer.Conv2d(256, 256, kernel_size=3, padding=1)
            self.bn6 = layer.BatchNorm2d(256)
            self.sn6 = neuron.LIFNode(v_threshold=self.th, tau=self.tau)
            self.conv7 = layer.Conv2d(256, 256, kernel_size=3, padding=1)
            self.bn7 = layer.BatchNorm2d(256)
            self.sn7 = neuron.LIFNode(v_threshold=self.th, tau=self.tau)


            self.flat = layer.Flatten()
            self.global_fc1 = layer.Linear(256 * 4 * 4, 1024)
            self.sn12 = neuron.LIFNode(v_threshold=self.th3, tau=self.tau)
            self.global_fc2 = layer.Linear(1024, args.num_classes)
            self.sn13 = neuron.LIFNode(v_threshold=self.th3, tau=self.tau)


        # 多步模式
        functional.set_step_mode(self, step_mode='m')

        # 初始化权重
        self._initialize_weights()

    # 初始化权重
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.Conv2d)):
                torch.nn.init.kaiming_normal_(m.weight.data)
                if m.bias is not None:
                    torch.nn.init.zeros_(m.bias.data)
            elif isinstance(m, nn.Linear):
                torch.nn.init.kaiming_normal_(m.weight.data)
                if m.bias is not None:
                    torch.nn.init.zeros_(m.bias.data)
            elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
                m.weight.data.fill_(1)
                torch.nn.init.zeros_(m.bias.data)




    def forward(self, x):
        # 添加时间步
        x = x.unsqueeze(0).repeat(self.T, 1, 1, 1, 1)

        x = self.sn1(self.bn1(self.conv1(x)))
        x = self.sn2(self.bn2(self.conv2(x)))
        x = self.sn3(self.bn3(self.conv3(x)))
        x = self.pool(x)

        if self.cplx == 1:

            # x = x.view(-1, 32 * 16 * 16)
            x = self.flat(x)
            x = self.sn8(self.fc1(x))
            x = self.sn11(self.fc(x))

            x = x.mean(0)
            return x

        x = self.sn4(self.bn4(self.conv4(x)))

        if self.cplx == 2:
            x = self.pool(x)
            # x = x.view(-1, 64 * 8 * 8)
            x = self.flat(x)
            x = self.sn9(self.fc2(x))
            x = self.sn11(self.fc(x))

            x = x.mean(0)
            return x

        x = self.sn5(self.bn5(self.conv5(x)))
        x = self.pool(x)

        if self.cplx == 3:
            # x = x.view(-1, 64 * 8 * 8)
            x = self.flat(x)
            x = self.sn9(self.fc2(x))
            x = self.sn11(self.fc(x))

            x = x.mean(0)
            return x

        x = self.sn6(self.bn6(self.conv6(x)))

        if self.cplx == 4:
            x = self.pool(x)
            # x = x.view(-1, 128 * 4 * 4)
            x = self.flat(x)
            x = self.sn10(self.fc3(x))
            x = self.sn11(self.fc(x))

            x = x.mean(0)
            return x

        x = self.sn7(self.bn7(self.conv6(x)))
        x = self.pool(x)

        if self.cplx == 5:
            # x = x.view(-1, 128 * 4 * 4)
            x = self.flat(x)
            x = self.sn12(self.global_fc1(x))
            x = self.sn13(self.global_fc2(x))

            x = x.mean(0)
            return x


class SNNFmnistComplexity(nn.Module):
    def __init__(self, args, complexity):
        super(SNNFmnistComplexity, self).__init__()
        self.T = args.T
        self.th = args.threshold
        self.th2 = args.threshold2
        self.th3 = args.threshold3
        self.cplx = complexity


        self.conv1 = layer.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1 = layer.BatchNorm2d(32)
        self.sn1 = neuron.IFNode(v_threshold=self.th)
        self.conv2 = layer.Conv2d(32, 32, kernel_size=3, padding=1)
        self.bn2 = layer.BatchNorm2d(32)
        self.sn2 = neuron.IFNode(v_threshold=self.th)

        self.pool = layer.MaxPool2d(kernel_size=2)
        # self.sn3 = neuron.IFNode()


        if self.cplx == 1:
            self.flat = layer.Flatten()

            self.fc1 = layer.Linear(32 * 14 * 14, 128)
            self.sn7 = neuron.IFNode(v_threshold=self.th)
            self.fc = layer.Linear(128, args.num_classes)
            self.sn10 = neuron.IFNode(v_threshold=self.th)

        elif self.cplx == 2:
            self.conv3 = layer.Conv2d(32, 64, kernel_size=3, padding=1)
            self.bn3 = layer.BatchNorm2d(64)
            self.sn3 = neuron.IFNode(v_threshold=self.th)

            self.flat = layer.Flatten()
            self.fc2 = layer.Linear(64 * 7 * 7, 128)
            self.sn8 = neuron.IFNode(v_threshold=self.th)
            self.fc = layer.Linear(128, args.num_classes)
            self.sn10 = neuron.IFNode(v_threshold=self.th)


        elif self.cplx == 3:
            self.conv3 = layer.Conv2d(32, 64, kernel_size=3, padding=1)
            self.bn3 = layer.BatchNorm2d(64)
            self.sn3 = neuron.IFNode(v_threshold=self.th)
            self.conv4 = layer.Conv2d(64, 64, kernel_size=3, padding=1)
            self.bn4 = layer.BatchNorm2d(64)
            self.sn4 = neuron.IFNode(v_threshold=self.th2)


            self.flat = layer.Flatten()
            self.fc2 = layer.Linear(64 * 7 * 7, 128)
            self.sn8 = neuron.IFNode(v_threshold=self.th2)
            self.fc = layer.Linear(128, args.num_classes)
            self.sn10 = neuron.IFNode(v_threshold=self.th2)


        elif self.cplx == 4:
            self.conv3 = layer.Conv2d(32, 64, kernel_size=3, padding=1)
            self.bn3 = layer.BatchNorm2d(64)
            self.sn3 = neuron.IFNode(v_threshold=self.th)
            self.conv4 = layer.Conv2d(64, 64, kernel_size=3, padding=1)
            self.bn4 = layer.BatchNorm2d(64)
            self.sn4 = neuron.IFNode(v_threshold=self.th2)


            self.conv5 = layer.Conv2d(64, 128, kernel_size=3, padding=1)
            self.bn5 = layer.BatchNorm2d(128)
            self.sn5 = neuron.IFNode(v_threshold=self.th3)

            self.flat = layer.Flatten()
            self.fc3 = layer.Linear(128 * 7 * 7, 128)
            self.sn9 = neuron.IFNode(v_threshold=self.th3)
            self.fc = layer.Linear(128, args.num_classes)
            self.sn10 = neuron.IFNode(v_threshold=self.th3)

        elif self.cplx == 5:
            self.conv3 = layer.Conv2d(32, 64, kernel_size=3, padding=1)
            self.bn3 = layer.BatchNorm2d(64)
            self.sn3 = neuron.IFNode(v_threshold=self.th)
            self.conv4 = layer.Conv2d(64, 64, kernel_size=3, padding=1)
            self.bn4 = layer.BatchNorm2d(64)
            self.sn4 = neuron.IFNode(v_threshold=self.th2)


            self.conv5 = layer.Conv2d(64, 128, kernel_size=3, padding=1)
            self.bn5 = layer.BatchNorm2d(128)
            self.sn5 = neuron.IFNode(v_threshold=self.th3)
            self.conv6 = layer.Conv2d(128, 128, kernel_size=3, padding=1)
            self.bn6 = layer.BatchNorm2d(128)
            self.sn6 = neuron.IFNode(v_threshold=self.th3)


            self.flat = layer.Flatten()
            self.global_fc1 = layer.Linear(128 * 7 * 7, 128)
            self.sn11 = neuron.IFNode(v_threshold=self.th3)
            self.global_fc2 = layer.Linear(128, args.num_classes)
            self.sn12 = neuron.IFNode(v_threshold=self.th3)


        # 多步模式
        functional.set_step_mode(self, step_mode='m')

        # 初始化权重
        self._initialize_weights()

    # 初始化权重
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.Conv2d)):
                torch.nn.init.kaiming_normal_(m.weight.data)
                if m.bias is not None:
                    torch.nn.init.zeros_(m.bias.data)
            elif isinstance(m, nn.Linear):
                torch.nn.init.kaiming_normal_(m.weight.data)
                if m.bias is not None:
                    torch.nn.init.zeros_(m.bias.data)
            elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
                m.weight.data.fill_(1)
                torch.nn.init.zeros_(m.bias.data)




    def forward(self, x):
        # 添加时间步
        x = x.unsqueeze(0).repeat(self.T, 1, 1, 1, 1)

        x = self.sn1(self.bn1(self.conv1(x)))
        x = self.sn2(self.bn2(self.conv2(x)))
        x = self.pool(x)

        if self.cplx == 1:

            # x = x.view(-1, 32 * 16 * 16)
            x = self.flat(x)
            x = self.sn7(self.fc1(x))
            x = self.sn10(self.fc(x))

            x = x.mean(0)
            return x

        x = self.sn3(self.bn3(self.conv3(x)))

        if self.cplx == 2:
            x = self.pool(x)
            # x = x.view(-1, 64 * 8 * 8)
            x = self.flat(x)
            x = self.sn8(self.fc2(x))
            x = self.sn10(self.fc(x))

            x = x.mean(0)
            return x

        x = self.sn4(self.bn4(self.conv4(x)))
        x = self.pool(x)

        if self.cplx == 3:
            # x = x.view(-1, 64 * 8 * 8)
            x = self.flat(x)
            x = self.sn8(self.fc2(x))
            x = self.sn10(self.fc(x))

            x = x.mean(0)
            return x

        x = self.sn5(self.bn5(self.conv5(x)))

        if self.cplx == 4:
            # x = self.pool(x)
            # x = x.view(-1, 128 * 4 * 4)
            x = self.flat(x)
            x = self.sn9(self.fc3(x))
            x = self.sn10(self.fc(x))

            x = x.mean(0)
            return x

        x = self.sn6(self.bn6(self.conv6(x)))
        # x = self.pool(x)

        if self.cplx == 5:
            # x = x.view(-1, 128 * 4 * 4)
            x = self.flat(x)
            x = self.sn11(self.global_fc1(x))
            x = self.sn12(self.global_fc2(x))

            x = x.mean(0)
            return x





class SNN_BNTT(nn.Module):
    def __init__(self, args, leak_mem=0.95, img_size=32,  num_cls=10):
        super(SNN_BNTT, self).__init__()

        self.img_size = img_size
        self.num_cls = num_cls
        self.timesteps = args.T
        self.spike_fn = Surrogate_BP_Function.apply
        self.leak_mem = leak_mem
        self.batch_num = self.timesteps

        # print (">>>>>>>>>>>>>>>>>>> VGG 9 >>>>>>>>>>>>>>>>>>>>>>")
        # print ("***** time step per batchnorm".format(self.batch_num))
        # print (">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")

        affine_flag = True
        bias_flag = False

        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=bias_flag)
        self.bntt1 = nn.ModuleList([nn.BatchNorm2d(32, eps=1e-4, momentum=0.1, affine=affine_flag) for i in range(self.batch_num)])
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, bias=bias_flag)
        self.bntt2 = nn.ModuleList([nn.BatchNorm2d(32, eps=1e-4, momentum=0.1, affine=affine_flag) for i in range(self.batch_num)])
        self.pool1 = nn.AvgPool2d(kernel_size=2)

        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1, bias=bias_flag)
        self.bntt3 = nn.ModuleList([nn.BatchNorm2d(64, eps=1e-4, momentum=0.1, affine=affine_flag) for i in range(self.batch_num)])
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=bias_flag)
        self.bntt4 = nn.ModuleList([nn.BatchNorm2d(64, eps=1e-4, momentum=0.1, affine=affine_flag) for i in range(self.batch_num)])
        self.pool2 = nn.AvgPool2d(kernel_size=2)

        self.conv5 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, bias=bias_flag)
        self.bntt5 = nn.ModuleList([nn.BatchNorm2d(128, eps=1e-4, momentum=0.1, affine=affine_flag) for i in range(self.batch_num)])
        self.conv6 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=bias_flag)
        self.bntt6 = nn.ModuleList([nn.BatchNorm2d(128, eps=1e-4, momentum=0.1, affine=affine_flag) for i in range(self.batch_num)])
        self.pool3 = nn.AvgPool2d(kernel_size=2)


        self.global_fc1 = nn.Linear(128 * 4 * 4, 128, bias=bias_flag)
        self.bntt_fc = nn.ModuleList([nn.BatchNorm1d(128, eps=1e-4, momentum=0.1, affine=affine_flag) for i in range(self.batch_num)])
        self.global_fc2 = nn.Linear(128, self.num_cls, bias=bias_flag)

        self.conv_list = [self.conv1, self.conv2, self.conv3, self.conv4, self.conv5, self.conv6]
        self.bntt_list = [self.bntt1, self.bntt2, self.bntt3, self.bntt4, self.bntt5, self.bntt6, self.bntt_fc]
        self.pool_list = [False, self.pool1, False, self.pool2, False, self.pool3]

        # Turn off bias of BNTT
        for bn_list in self.bntt_list:
            for bn_temp in bn_list:
                bn_temp.bias = None


        # Initialize the firing thresholds of all the layers
        for m in self.modules():
            if (isinstance(m, nn.Conv2d)):
                m.threshold = 1.0
                torch.nn.init.xavier_uniform_(m.weight, gain=2)
            elif (isinstance(m, nn.Linear)):
                m.threshold = 1.0
                torch.nn.init.xavier_uniform_(m.weight, gain=2)




    def forward(self, inp):

        batch_size = inp.size(0)
        mem_conv1 = torch.zeros(batch_size, 32, self.img_size, self.img_size).cuda()
        mem_conv2 = torch.zeros(batch_size, 32, self.img_size, self.img_size).cuda()
        mem_conv3 = torch.zeros(batch_size, 64, self.img_size//2, self.img_size//2).cuda()
        mem_conv4 = torch.zeros(batch_size, 64, self.img_size//2, self.img_size//2).cuda()
        mem_conv5 = torch.zeros(batch_size, 128, self.img_size//4, self.img_size//4).cuda()
        mem_conv6 = torch.zeros(batch_size, 128, self.img_size//4, self.img_size//4).cuda()
        mem_conv_list = [mem_conv1, mem_conv2, mem_conv3, mem_conv4, mem_conv5, mem_conv6]

        mem_fc1 = torch.zeros(batch_size, 128).cuda()
        mem_fc2 = torch.zeros(batch_size, self.num_cls).cuda()



        for t in range(self.timesteps):

            spike_inp = PoissonGen(inp)
            out_prev = spike_inp

            for i in range(len(self.conv_list)):
                mem_conv_list[i] = self.leak_mem * mem_conv_list[i] + self.bntt_list[i][t](self.conv_list[i](out_prev))
                mem_thr = (mem_conv_list[i] / self.conv_list[i].threshold) - 1.0
                out = self.spike_fn(mem_thr)
                rst = torch.zeros_like(mem_conv_list[i]).cuda()
                rst[mem_thr > 0] = self.conv_list[i].threshold
                mem_conv_list[i] = mem_conv_list[i] - rst
                out_prev = out.clone()


                if self.pool_list[i] is not False:
                    out = self.pool_list[i](out_prev)
                    out_prev = out.clone()


            out_prev = out_prev.reshape(batch_size, -1)

            if batch_size == 1:
                mem_fc1_x = self.global_fc1(out_prev)
            else:
                mem_fc1_x = self.bntt_fc[t](self.global_fc1(out_prev))
            mem_fc1 = self.leak_mem * mem_fc1 + mem_fc1_x

            #mem_fc1 = self.leak_mem * mem_fc1 + self.bntt_fc[t](self.fc1(out_prev))
            mem_thr = (mem_fc1 / self.global_fc1.threshold) - 1.0
            out = self.spike_fn(mem_thr)
            rst = torch.zeros_like(mem_fc1).cuda()
            rst[mem_thr > 0] = self.global_fc1.threshold
            mem_fc1 = mem_fc1 - rst
            out_prev = out.clone()

            # accumulate voltage in the last layer
            mem_fc2 = mem_fc2 + self.global_fc2(out_prev)

        out_voltage = mem_fc2 / self.timesteps


        return out_voltage



class SNN_complexity_BNTT(nn.Module):
    def __init__(self,  args, complexity, leak_mem=0.95, img_size=32,  num_cls=10):
        super(SNN_complexity_BNTT, self).__init__()

        self.timesteps = args.T


        self.img_size = img_size
        self.num_cls = num_cls

        self.spike_fn = Surrogate_BP_Function.apply
        self.leak_mem = leak_mem
        self.batch_num = self.timesteps




        #分层
        self.cplx = complexity

        # print(">>>>>>>>>>>>>>>>>>> VGG 9 >>>>>>>>>>>>>>>>>>>>>>")
        # print("***** time step per batchnorm".format(self.batch_num))
        # print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")

        affine_flag = True
        bias_flag = False

        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=bias_flag)
        self.bntt1 = nn.ModuleList([nn.BatchNorm2d(32, eps=1e-4, momentum=0.1, affine=affine_flag) for i in range(self.batch_num)])
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, bias=bias_flag)
        self.bntt2 = nn.ModuleList([nn.BatchNorm2d(32, eps=1e-4, momentum=0.1, affine=affine_flag) for i in range(self.batch_num)])
        self.pool1 = nn.MaxPool2d(kernel_size=2)



        if self.cplx == 1:
            print('模型1')
            self.fc1 = nn.Linear(32 * 16 * 16, 128, bias=bias_flag)
            self.bntt_fc = nn.ModuleList(
                [nn.BatchNorm1d(128, eps=1e-4, momentum=0.1, affine=affine_flag) for i in range(self.batch_num)])
            self.fc = nn.Linear(128, self.num_cls, bias=bias_flag)

        elif self.cplx == 2:
            print('模型2')
            self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1, bias=bias_flag)
            self.bntt3 = nn.ModuleList([nn.BatchNorm2d(64, eps=1e-4, momentum=0.1, affine=affine_flag) for i in range(self.batch_num)])
            self.pool2 = nn.MaxPool2d(kernel_size=2)
            self.fc2 = nn.Linear(64 * 8 * 8, 128, bias=bias_flag)
            self.bntt_fc = nn.ModuleList(
                [nn.BatchNorm1d(128, eps=1e-4, momentum=0.1, affine=affine_flag) for i in range(self.batch_num)])
            self.fc = nn.Linear(128, self.num_cls, bias=bias_flag)

        elif self.cplx == 3:
            print('模型3')
            self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1, bias=bias_flag)
            self.bntt3 = nn.ModuleList(
                [nn.BatchNorm2d(64, eps=1e-4, momentum=0.1, affine=affine_flag) for i in range(self.batch_num)])
            self.conv4 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=bias_flag)
            self.bntt4 = nn.ModuleList(
                [nn.BatchNorm2d(64, eps=1e-4, momentum=0.1, affine=affine_flag) for i in range(self.batch_num)])
            self.pool2 = nn.MaxPool2d(kernel_size=2)
            self.fc3 = nn.Linear(64 * 8 * 8, 128, bias=bias_flag)
            self.bntt_fc = nn.ModuleList(
                [nn.BatchNorm1d(128, eps=1e-4, momentum=0.1, affine=affine_flag) for i in range(self.batch_num)])
            self.fc = nn.Linear(128, self.num_cls, bias=bias_flag)

        elif self.cplx == 4:
            print('模型4')
            self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1, bias=bias_flag)
            self.bntt3 = nn.ModuleList(
                [nn.BatchNorm2d(64, eps=1e-4, momentum=0.1, affine=affine_flag) for i in range(self.batch_num)])
            self.conv4 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=bias_flag)
            self.bntt4 = nn.ModuleList(
                [nn.BatchNorm2d(64, eps=1e-4, momentum=0.1, affine=affine_flag) for i in range(self.batch_num)])
            self.pool2 = nn.MaxPool2d(kernel_size=2)
            self.conv5 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, bias=bias_flag)
            self.bntt5 = nn.ModuleList(
                [nn.BatchNorm2d(128, eps=1e-4, momentum=0.1, affine=affine_flag) for i in range(self.batch_num)])
            self.pool3 = nn.MaxPool2d(kernel_size=2)
            self.fc4 = nn.Linear(128 * 4 * 4, 128, bias=bias_flag)
            self.bntt_fc = nn.ModuleList(
                [nn.BatchNorm1d(128, eps=1e-4, momentum=0.1, affine=affine_flag) for i in range(self.batch_num)])
            self.fc = nn.Linear(128, self.num_cls, bias=bias_flag)

        elif self.cplx == 5:
            print('模型5')
            self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1, bias=bias_flag)
            self.bntt3 = nn.ModuleList(
                [nn.BatchNorm2d(64, eps=1e-4, momentum=0.1, affine=affine_flag) for i in range(self.batch_num)])
            self.conv4 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=bias_flag)
            self.bntt4 = nn.ModuleList(
                [nn.BatchNorm2d(64, eps=1e-4, momentum=0.1, affine=affine_flag) for i in range(self.batch_num)])
            self.pool2 = nn.MaxPool2d(kernel_size=2)
            self.conv5 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, bias=bias_flag)
            self.bntt5 = nn.ModuleList(
                [nn.BatchNorm2d(128, eps=1e-4, momentum=0.1, affine=affine_flag) for i in range(self.batch_num)])
            self.conv6 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=bias_flag)
            self.bntt6 = nn.ModuleList(
                [nn.BatchNorm2d(128, eps=1e-4, momentum=0.1, affine=affine_flag) for i in range(self.batch_num)])
            self.pool3 = nn.MaxPool2d(kernel_size=2)
            self.global_fc1 = nn.Linear(128 * 4 * 4, 128, bias=bias_flag)
            self.bntt_fc = nn.ModuleList([nn.BatchNorm1d(128, eps=1e-4, momentum=0.1, affine=affine_flag) for i in range(self.batch_num)])
            self.global_fc2 = nn.Linear(128, self.num_cls, bias=bias_flag)



        if self.cplx == 1:
            self.conv_list = [self.conv1, self.conv2]
            self.bntt_list = [self.bntt1, self.bntt2, self.bntt_fc]
            self.pool_list = [False, self.pool1]

        elif self.cplx == 2:
            self.conv_list = [self.conv1, self.conv2, self.conv3]
            self.bntt_list = [self.bntt1, self.bntt2, self.bntt3, self.bntt_fc]
            self.pool_list = [False, self.pool1, self.pool2]

        elif self.cplx == 3:
            self.conv_list = [self.conv1, self.conv2, self.conv3, self.conv4]
            self.bntt_list = [self.bntt1, self.bntt2, self.bntt3, self.bntt4, self.bntt_fc]
            self.pool_list = [False, self.pool1, False, self.pool2]

        elif self.cplx == 4:
            self.conv_list = [self.conv1, self.conv2, self.conv3, self.conv4, self.conv5]
            self.bntt_list = [self.bntt1, self.bntt2, self.bntt3, self.bntt4, self.bntt5, self.bntt_fc]
            self.pool_list = [False, self.pool1, False, self.pool2, self.pool3]

        elif self.cplx == 5:
            self.conv_list = [self.conv1, self.conv2, self.conv3, self.conv4, self.conv5, self.conv6]
            self.bntt_list = [self.bntt1, self.bntt2, self.bntt3, self.bntt4, self.bntt5, self.bntt6, self.bntt_fc]
            self.pool_list = [False, self.pool1, False, self.pool2, False, self.pool3]

        # Turn off bias of BNTT
        for bn_list in self.bntt_list:
            for bn_temp in bn_list:
                bn_temp.bias = None


        # Initialize the firing thresholds of all the layers
        for m in self.modules():
            if (isinstance(m, nn.Conv2d)):
                m.threshold = 1.0
                torch.nn.init.xavier_uniform_(m.weight, gain=2)
            elif (isinstance(m, nn.Linear)):
                m.threshold = 1.0
                torch.nn.init.xavier_uniform_(m.weight, gain=2)




    def forward(self, inp):

        batch_size = inp.size(0)

        mem_conv1 = torch.zeros(batch_size, 32, self.img_size, self.img_size).cuda()
        mem_conv2 = torch.zeros(batch_size, 32, self.img_size, self.img_size).cuda()
        mem_conv3 = torch.zeros(batch_size, 64, self.img_size//2, self.img_size//2).cuda()
        mem_conv4 = torch.zeros(batch_size, 64, self.img_size//2, self.img_size//2).cuda()
        mem_conv5 = torch.zeros(batch_size, 128, self.img_size//4, self.img_size//4).cuda()
        mem_conv6 = torch.zeros(batch_size, 128, self.img_size//4, self.img_size//4).cuda()
        mem_conv_list = [mem_conv1, mem_conv2, mem_conv3, mem_conv4, mem_conv5, mem_conv6]

        mem_fc1 = torch.zeros(batch_size, 128).cuda()
        mem_fc2 = torch.zeros(batch_size, self.num_cls).cuda()

        for t in range(self.timesteps):
            spike_inp = PoissonGen(inp)
            out_prev = spike_inp

            for i in range(len(self.conv_list)):
                mem_conv_list[i] = self.leak_mem * mem_conv_list[i] + self.bntt_list[i][t](self.conv_list[i](out_prev))
                # V[t] = \tau_m * V[t-1] + X[t]
                mem_thr = (mem_conv_list[i] / self.conv_list[i].threshold) - 1.0
                #   TODO: mem*threshold = V[t] - 1*threshold   mem = V[t]/threshold - 1
                out = self.spike_fn(mem_thr)
                rst = torch.zeros_like(mem_conv_list[i]).cuda()
                rst[mem_thr > 0] = self.conv_list[i].threshold
                mem_conv_list[i] = mem_conv_list[i] - rst  # soft reset
                out_prev = out.clone()


                if self.pool_list[i] is not False:
                    out = self.pool_list[i](out_prev)
                    out_prev = out.clone()

            out_prev = out_prev.reshape(batch_size, -1)

            if self.cplx == 1:
                if batch_size == 1:
                    mem_fc1_x = self.fc1(out_prev)
                else:
                    mem_fc1_x = self.bntt_fc[t](self.fc1(out_prev))
                mem_fc1 = self.leak_mem * mem_fc1 + mem_fc1_x
                mem_thr = (mem_fc1 / self.fc1.threshold) - 1.0
                out = self.spike_fn(mem_thr)
                rst = torch.zeros_like(mem_fc1).cuda()
                rst[mem_thr > 0] = self.fc1.threshold
                mem_fc1 = mem_fc1 - rst
                out_prev = out.clone()

                # accumulate voltage in the last layer
                mem_fc2 = mem_fc2 + self.fc(out_prev)
            elif self.cplx == 2:
                if batch_size == 1:
                    mem_fc1_x = self.fc2(out_prev)
                else:
                    mem_fc1_x = self.bntt_fc[t](self.fc2(out_prev))
                mem_fc1 = self.leak_mem * mem_fc1 + mem_fc1_x
                mem_thr = (mem_fc1 / self.fc2.threshold) - 1.0
                out = self.spike_fn(mem_thr)
                rst = torch.zeros_like(mem_fc1).cuda()
                rst[mem_thr > 0] = self.fc2.threshold
                mem_fc1 = mem_fc1 - rst
                out_prev = out.clone()

                # accumulate voltage in the last layer
                mem_fc2 = mem_fc2 + self.fc(out_prev)
            elif self.cplx == 3:
                if batch_size == 1:
                    mem_fc1_x = self.fc3(out_prev)
                else:
                    mem_fc1_x = self.bntt_fc[t](self.fc3(out_prev))
                mem_fc1 = self.leak_mem * mem_fc1 + mem_fc1_x
                mem_thr = (mem_fc1 / self.fc3.threshold) - 1.0
                out = self.spike_fn(mem_thr)
                rst = torch.zeros_like(mem_fc1).cuda()
                rst[mem_thr > 0] = self.fc3.threshold
                mem_fc1 = mem_fc1 - rst
                out_prev = out.clone()

                # accumulate voltage in the last layer
                mem_fc2 = mem_fc2 + self.fc(out_prev)
            elif self.cplx == 4:
                if batch_size == 1:
                    mem_fc1_x = self.fc4(out_prev)
                else:
                    mem_fc1_x = self.bntt_fc[t](self.fc4(out_prev))
                mem_fc1 = self.leak_mem * mem_fc1 + mem_fc1_x
                mem_thr = (mem_fc1 / self.fc4.threshold) - 1.0
                out = self.spike_fn(mem_thr)
                rst = torch.zeros_like(mem_fc1).cuda()
                rst[mem_thr > 0] = self.fc4.threshold
                mem_fc1 = mem_fc1 - rst
                out_prev = out.clone()

                # accumulate voltage in the last layer
                mem_fc2 = mem_fc2 + self.fc(out_prev)
            elif self.cplx == 5:
                if batch_size == 1:
                    mem_fc1_x = self.global_fc1(out_prev)
                else:
                    mem_fc1_x = self.bntt_fc[t](self.global_fc1(out_prev))
                mem_fc1 = self.leak_mem * mem_fc1 + mem_fc1_x
                mem_thr = (mem_fc1 / self.global_fc1.threshold) - 1.0
                out = self.spike_fn(mem_thr)
                rst = torch.zeros_like(mem_fc1).cuda()
                rst[mem_thr > 0] = self.global_fc1.threshold
                mem_fc1 = mem_fc1 - rst
                out_prev = out.clone()

                # accumulate voltage in the last layer
                mem_fc2 = mem_fc2 + self.global_fc2(out_prev)

            # mem_fc1 = self.leak_mem * mem_fc1 + self.bntt_fc[t](self.fc1(out_prev))
            # mem_thr = (mem_fc1 / self.fc1.threshold) - 1.0
            # out = self.spike_fn(mem_thr)
            # rst = torch.zeros_like(mem_fc1).cuda()
            # rst[mem_thr > 0] = self.fc1.threshold
            # mem_fc1 = mem_fc1 - rst
            # out_prev = out.clone()

            # accumulate voltage in the last layer
            # mem_fc2 = mem_fc2 + self.fc(out_prev)

        out_voltage = mem_fc2 / self.timesteps


        return out_voltage











#模型分配概率分布矩阵
def init_cplx_dict_snn(args, flag):  # flag is the preference for complexity
    idx_cplx_dict = {}               # {idxs_user: complexity  -->  0: 5, 1: 1, 2: 4, ...}
    prob = [[0.6, 0.1, 0.1, 0.1, 0.1],
            [0.1, 0.6, 0.1, 0.1, 0.1],
            [0.1, 0.1, 0.6, 0.1, 0.1],
            [0.1, 0.1, 0.1, 0.6, 0.1],
            [0.1, 0.1, 0.1, 0.1, 0.6]]
    # 分配客户端规模
    if flag == 6:
        for i in (0, 1):
            idx_cplx_dict[i] = int(1)
        for i in (2, 3):
            idx_cplx_dict[i] = int(2)
        for i in (4, 5):
            idx_cplx_dict[i] = int(3)
        for i in (6, 7):
            idx_cplx_dict[i] = int(4)
        for i in (8, 9):
            idx_cplx_dict[i] = int(5)
    else:
        for i in range(args.num_users):
            idx_cplx_dict[i] = int( np.random.choice([1, 2, 3, 4, 5], 1, p=prob[flag-1]) )
    return idx_cplx_dict

def download_hetero_net_snn(global_model, local_model, cplx):   # download
    local_update = copy.deepcopy(local_model.state_dict())
    if cplx == 5:
        for name in local_update.keys():
            # print(global_model.state_dict().keys())
            local_update[name] = global_model.state_dict()[name]
        local_model.load_state_dict(local_update)
    else:
        for name in local_update.keys():
            if 'fc' in name:
                continue
            else:
                local_update[name] = global_model.state_dict()[name]
        local_model.load_state_dict(local_update)
    return local_model

def generate_cplx_net_snn(args, global_model, user_idx, idx_cplx_dict, device='cuda'):   # download
    cplx = idx_cplx_dict[user_idx]
    local_model = SNNCifarComplexity(args, cplx).to(device)
    local_update = copy.deepcopy(local_model.state_dict())
    if cplx == 5:
        for name in local_update.keys():
            local_update[name] = global_model.state_dict()[name]
        local_model.load_state_dict(local_update)
    else:
        for name in local_update.keys():
            if 'fc' in name:
                continue
            else:
                local_update[name] = global_model.state_dict()[name]
        local_model.load_state_dict(local_update)
    return local_model


if __name__ == "__main__":
    args = args_parser()
    # input = torch.randn(1, 3, 32, 32)
    for i in [1, 2, 3, 4, 5]:
        print('--------------------------------------------------------------------------------')
        # model = EnhancedSNNCifar(args)
        model = SNNVGG9Complexity(args, i)
        print(model)
        input_shape = (1, 3, 32, 32)
        # summary(model, input_shape)
        # macs, params = profile(model, inputs=(input, ))


