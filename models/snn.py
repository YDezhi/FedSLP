from spikingjelly.activation_based import neuron, layer
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


class EnhancedSNNCifar(nn.Module):
    def __init__(self, args):
        super(EnhancedSNNCifar, self).__init__()
        self.conv1 = layer.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = layer.BatchNorm2d(32)
        self.conv2 = layer.Conv2d(32, 32, kernel_size=3, padding=1)
        self.bn2 = layer.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(kernel_size=2)

        self.conv3 = layer.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn3 = layer.BatchNorm2d(64)
        self.conv4 = layer.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn4 = layer.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(kernel_size=2)

        self.conv5 = layer.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn5 = layer.BatchNorm2d(128)
        self.conv6 = layer.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn6 = layer.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(kernel_size=2)

        self.global_fc1 = layer.Linear(128 * 4 * 4, 128)
        self.global_fc2 = layer.Linear(128, args.num_classes)

    def forward(self, x):
        x = self.bn1(neuron.IFNode.apply(self.conv1(x)))
        x = self.bn2(neuron.IFNode.apply(self.conv2(x)))
        x = self.pool1(x)

        x = self.bn3(neuron.IFNode.apply(self.conv3(x)))
        x = self.bn4(neuron.IFNode.apply(self.conv4(x)))
        x = self.pool2(x)

        x = self.bn5(neuron.IFNode.apply(self.conv5(x)))
        x = self.bn6(neuron.IFNode.apply(self.conv6(x)))
        x = self.pool3(x)

        x = x.view(-1, 128 * 4 * 4)

        x = self.global_fc1(x)
        x = self.global_fc2(x)

        return x

    # def forward(self, x):
    #     x = neuron.neuron_fn(self.bn1(self.conv1(x)), 'IF')
    #     x = neuron.neuron_fn(self.bn2(self.conv2(x)), 'IF')
    #     x = self.pool1(x)
    #
    #     x = neuron.neuron_fn(self.bn3(self.conv3(x)), 'IF')
    #     x = neuron.neuron_fn(self.bn4(self.conv4(x)), 'IF')
    #     x = self.pool2(x)
    #
    #     x = neuron.neuron_fn(self.bn5(self.conv5(x)), 'IF')
    #     x = neuron.neuron_fn(self.bn6(self.conv6(x)), 'IF')
    #     x = self.pool3(x)
    #
    #     x = x.view(-1, 128 * 4 * 4)
    #
    #     x = self.global_fc1(x)
    #     x = self.global_fc2(x)
    #
    #     return x



class SNNCifarComplexity(nn.Module):
    def __init__(self, args, complexity):
        super(SNNCifarComplexity, self).__init__()

        self.cplx = complexity

        self.conv1 = layer.Conv2d(2, 32, kernel_size=3, padding=1)
        sn1 = neuron.IFNode()
        self.bn1 = layer.BatchNorm2d(32)
        self.conv2 = layer.Conv2d(32, 32, kernel_size=3, padding=1)
        self.sn2 = neuron.IFNode()
        self.bn2 = layer.BatchNorm2d(32)
        self.pool = nn.MaxPool2d(kernel_size=2)
        # self.sn3 = neuron.IFNode()
        if self.cplx == 1:
            self.fc1 = layer.Linear(32 * 16 * 16, 128)
            self.fc = layer.Linear(128, args.num_classes)

        elif self.cplx == 2:
            self.conv3 = layer.Conv2d(32, 64, kernel_size=3, padding=1)
            self.sn3 = neuron.IFNode()
            self.bn3 = layer.BatchNorm2d(64)
            self.fc2 = layer.Linear(64 * 8 * 8, 128)
            self.fc = layer.Linear(128, args.num_classes)

        elif self.cplx == 3:
            self.conv3 = layer.Conv2d(32, 64, kernel_size=3, padding=1)
            self.sn3 = neuron.IFNode()
            self.bn3 = layer.BatchNorm2d(64)
            self.conv4 = layer.Conv2d(64, 64, kernel_size=3, padding=1)
            self.sn4 = neuron.IFNode()
            self.bn4 = layer.BatchNorm2d(64)
            self.fc2 = layer.Linear(64 * 8 * 8, 128)
            self.fc = layer.Linear(128, args.num_classes)

        elif self.cplx == 4:
            self.conv3 = layer.Conv2d(32, 64, kernel_size=3, padding=1)
            self.sn3 = neuron.IFNode()
            self.bn3 = layer.BatchNorm2d(64)
            self.conv4 = layer.Conv2d(64, 64, kernel_size=3, padding=1)
            self.sn4 = neuron.IFNode()
            self.bn4 = layer.BatchNorm2d(64)
            self.conv5 = layer.Conv2d(64, 128, kernel_size=3, padding=1)
            self.sn5 = neuron.IFNode()
            self.bn5 = layer.BatchNorm2d(128)
            self.fc3 = layer.Linear(128 * 4 * 4, 128)
            self.fc = layer.Linear(128, args.num_classes)

        elif self.cplx == 5:
            self.conv3 = layer.Conv2d(32, 64, kernel_size=3, padding=1)
            self.sn3 = neuron.IFNode()
            self.bn3 = layer.BatchNorm2d(64)
            self.conv4 = layer.Conv2d(64, 64, kernel_size=3, padding=1)
            self.sn4 = neuron.IFNode()
            self.bn4 = layer.BatchNorm2d(64)
            self.conv5 = layer.Conv2d(64, 128, kernel_size=3, padding=1)
            self.sn5 = neuron.IFNode()
            self.bn5 = layer.BatchNorm2d(128)
            self.conv6 = layer.Conv2d(128, 128, kernel_size=3, padding=1)
            self.sn6 = neuron.IFNode()
            self.bn6 = layer.BatchNorm2d(128)
            self.global_fc1 = layer.Linear(128 * 4 * 4, 128)
            self.global_fc2 = layer.Linear(128, args.num_classes)


    def forward(self, x):
        x = self.bn1(neuron.IFNode.apply(self.conv1(x)))
        x = self.bn2(neuron.IFNode.apply(self.conv2(x)))
        x = self.pool(x)

        if self.cplx == 1:
            x = x.view(-1, 32 * 16 * 16)
            x = self.fc1(x)
            x = self.fc(x)
            return x

        x = self.bn3(self.sn3(self.conv3(x)))

        if self.cplx == 2:
            x = self.pool(x)
            x = x.view(-1, 64 * 8 * 8)
            x = self.fc2(x)
            x = self.fc(x)
            return x

        x = self.bn4(self.sn4(self.conv4(x)))
        x = self.pool(x)

        if self.cplx == 3:
            x = x.view(-1, 64 * 8 * 8)
            x = self.fc2(x)
            x = self.fc(x)
            return x

        x = self.bn5(self.sn5(self.conv5(x)))

        if self.cplx == 4:
            x = self.pool(x)
            x = x.view(-1, 128 * 4 * 4)
            x = self.fc3(x)
            x = self.fc(x)
            return x

        x = self.bn6(self.sn6(self.conv6(x)))
        x = self.pool(x)

        if self.cplx == 5:
            x = x.view(-1, 128 * 4 * 4)
            x = self.global_fc1(x)
            x = self.global_fc2(x)
            return x


def init_cplx_dict(args, flag):  # flag is the preference for complexity
    idx_cplx_dict = {}                     # {idxs_user: complexity  -->  0: 5, 1: 1, 2: 4, ...}
    prob = [[0.6, 0.1, 0.1, 0.1, 0.1],
            [0.1, 0.6, 0.1, 0.1, 0.1],
            [0.1, 0.1, 0.6, 0.1, 0.1],
            [0.1, 0.1, 0.1, 0.6, 0.1],
            [0.1, 0.1, 0.1, 0.1, 0.6]]
    for i in range(args.num_users):
        idx_cplx_dict[i] = int( np.random.choice([1, 2, 3, 4, 5], 1, p=prob[flag-1]) )
    return idx_cplx_dict

def download_hetero_net(global_model, local_model, cplx):   # download
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

def generate_cplx_net(args, global_model, user_idx, idx_cplx_dict, device='cuda'):   # download
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
        model = SNNCifarComplexity(args, i)
        print(model)
        input_shape = (1, 3, 32, 32)
        # summary(model, input_shape)
        # macs, params = profile(model, inputs=(input, ))


# def count_macs_params(model, input_shape):
#     macs = 0
#     params = 0
#
#     # 遍历模型的每个层
#     for module in model.modules():
#         if isinstance(module, (torch.nn.Conv2d, torch.nn.Linear)):
#             # 计算卷积层和线性层的MACs和参数数量
#             weight_ops = module.weight.numel()
#             if module.bias is not None:
#                 bias_ops = module.bias.numel()
#             else:
#                 bias_ops = 0
#             macs += weight_ops
#             macs += bias_ops
#
#             params += weight_ops
#             params += bias_ops
#
#     return macs, params
#
# # 创建模型和输入数据
# args = args_parser()
# model = EnhancedSNNCifar(args)
# input_shape = (1, 3, 32, 32)
# input_data = torch.randn(input_shape)
#
# # 计算MACs和参数数量
# macs, params = count_macs_params(model, input_shape)
#
# print("MACs:", macs)
# print("Params:", params)