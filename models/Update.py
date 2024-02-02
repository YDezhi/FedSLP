#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import torch
from torch import nn, autograd
from torch.utils.data import DataLoader, Dataset
import numpy as np
import random
import copy
import torch.autograd as autograd
from spikingjelly.activation_based import functional, monitor, neuron
import torch.nn.functional as F
from utils.options import args_parser, details
from models.models_complexity import EnhancedCNNCifar, CNNCifarComplexity, init_cplx_dict, SmallCNNCifar, \
    EnhancedCNNFmnist, SmallCNNFmnist
from models.snn_complexity import EnhancedSNNCifar, SNNCifarComplexity, init_cplx_dict_snn, download_hetero_net_snn, SmallSNNCifar,SNN_BNTT
from models.test import test_img
from torchvision import datasets, transforms
import matplotlib.pyplot as plt








trans_cifar = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
dataset_train = datasets.CIFAR10('./data/cifar', train=True, download=True, transform=trans_cifar)
dataset_test = datasets.CIFAR10('./data/cifar', train=False, download=True, transform=trans_cifar)




class DatasetSplit(Dataset):            # Define __getitem__() method, so that we can fetch specific data in the dataset.
    def __init__(self, dataset, idxs):  # Here 'idxs' are ndarray.
        self.dataset = dataset
        self.idxs = list(idxs)

    def __len__(self):                  # Define the length of the class, control when the iterate in line 41 stops.
        return len(self.idxs)

    def __getitem__(self, item):        # idxs contain indexs of data in label order in dataset, length of 600 per list.
        image, label = self.dataset[self.idxs[item]]
        return image, label


class LocalUpdate(object):
    def __init__(self, args, dataset=None, idxs=None, tag=0, idx=0):
        self.args = args


        # 更改损失函数
        # self.loss_func = nn.MSELoss()
        #原损失函数
        self.loss_func = nn.CrossEntropyLoss()
        self.selected_clients = []
        self.ldr_train = DataLoader(DatasetSplit(dataset, idxs), batch_size=self.args.batchsize, shuffle=True)
        self.tag = tag
        self.idxs = idxs
        self.idx = idx

    def train(self, net):
        net.train()

        acc_test = []

        # train and update
        # optimizer = torch.optim.SGD(net.parameters(), lr=self.args.lr, momentum=self.args.momentum)
        if self.args.optimizer == 'sgd':
            optimizer = torch.optim.SGD(net.parameters(), lr=self.args.lr, momentum=self.args.momentum)
        elif self.args.optimizer == 'adam':
            optimizer = torch.optim.Adam(net.parameters(), lr=0.001)

        epoch_loss = []
        for iter in range(self.args.epoch):
            batch_loss = []
            for batch_idx, (images, labels) in enumerate(self.ldr_train):   # Iterate in the order of __getitem__(item)
                images, labels = images.to(self.args.device), labels.to(self.args.device)
                # print("idx is: ", self.idx, " before training: ", images.shape, labels.shape)
                net.zero_grad()

                log_probs = net(images)
                # except ValueError as e:
                #     print('报错！模型参数：')
                #     for name, param in net.named_parameters():
                #         print('名字：{}, 形状：{}, 参数：{}'.format(name, param.shape, param))



                #报错修改
                if self.args.model in ['snn', 'smallsnn', 'snnandann', 'smallsnnandann', 'snnbntt', 'snnvgg9']:
                    # labels = labels.float()
                    labels = F.one_hot(labels, 10).float()

                # 输出结果和预测标签

                loss = self.loss_func(log_probs, labels)

                # 报错更改 https://blog.csdn.net/qq_40737596/article/details/127674436
                # loss.requires_grad_(True)
                # 报错更改 retain_graph=True
                # torch.autograd.set_detect_anomaly(True)
                loss.backward()
                optimizer.step()
                if self.args.verbose and batch_idx % 10 == 0:
                    print('Update Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        iter, batch_idx * len(images), len(self.ldr_train.dataset),
                               100. * batch_idx / len(self.ldr_train), loss.item()))
                batch_loss.append(loss.item())
                # 如果使用snn就要重置网络
                if self.args.model in ['snn', 'smallsnn', 'snnandann', 'smallsnnandann', 'SNN_VGG9', 'snnvgg9'] and self.tag == 1:
                    functional.reset_net(net)




                # # 监视器
                # # 监视点火率函数
                # def cal_firing_rate(s_seq: torch.Tensor):
                #     # s_seq.shape = [T, N, *]
                #     return s_seq.flatten(1).mean(1)
                #
                # # 设置监视器
                # fr_monitor = monitor.OutputMonitor(net, neuron.IFNode, cal_firing_rate)
                # # 监视
                # with torch.no_grad():
                #     # functional.reset_net(net)
                #     # fr_monitor.disable()
                #     # net(images)
                #     functional.reset_net(net)
                #     # print(f'after call fr_monitor.disable(), fr_monitor.records=\n{fr_monitor.records}')
                #     fr_monitor.enable()
                #     a = net(images)
                #     print(a)
                #     print(f'after call fr_monitor.enable(), fr_monitor.records=\n{fr_monitor.records}')
                #     functional.reset_net(net)
                #     del fr_monitor


            # # 打印准确率
            # if self.args.model == 'smallsnn':
            #     a = SmallSNNCifar(args=self.args).to(self.args.device)
            # elif self.args.model == 'smallcnn' and self.args.dataset == 'cifar':
            #     a = SmallCNNCifar(args=self.args).to(self.args.device)
            # elif self.args.model == 'smallcnn' and self.args.dataset == 'fmnist':
            #     a = SmallCNNFmnist(args=self.args).to(self.args.device)
            # elif self.args.model == 'snn' and self.args.dataset == 'cifar':
            #     a = EnhancedSNNCifar(args=self.args).to(self.args.device)
            # elif self.args.model == 'snnbntt' and self.args.dataset == 'cifar':
            #     a = SNN_BNTT(args=self.args).to(self.args.device)
            # a.load_state_dict(net.state_dict())
            # # print('client {:3d}, loss {:.3f}'.format(idx, loss))
            # acc, loss = test_img(a, dataset_test, self.args)
            # print("client {} accuracy: {:.2f} loss : {:.3f}".format(self.idx, acc, loss))
            # #



            epoch_loss.append(sum(batch_loss)/len(batch_loss))
            # #
            # # acc_test.append(acc)

        # 监视器
        # 监视点火率函数
        # def cal_firing_rate(s_seq: torch.Tensor):
        #     # s_seq.shape = [T, N, *]
        #     return s_seq.flatten(1).mean(1)
        #
        # # 设置监视器
        # fr_monitor = monitor.OutputMonitor(net, neuron.IFNode, cal_firing_rate)
        # # 监视
        # with torch.no_grad():
        #     # functional.reset_net(net)
        #     # fr_monitor.disable()
        #     # net(images)
        #     # functional.reset_net(net)
        #     # print(f'after call fr_monitor.disable(), fr_monitor.records=\n{fr_monitor.records}')
        #     fr_monitor.enable()
        #     a = net(images)
        #     # print(a)
        #     print(f'after call fr_monitor.enable(), fr_monitor.records=\n{fr_monitor.records}')
        #     functional.reset_net(net)
        #     del fr_monitor


        # 存储以及画每个客户端本地训练时的aCC
        # acc_test = np.array(acc_test, dtype=float)
        # plt.figure()
        # plt.plot(range(len(acc_test)), acc_test)
        # plt.title('Test Accuracy vs. Communication Rounds')
        # plt.ylabel('Test Accuracy')
        # plt.xlabel('Communication Rounds')
        # plt.savefig('./save/fed_client{}_acc[{}].png'
        #             .format(self.idx, acc))
        #
        # print('··························一个客户端结束了·······················')


        # main_fed.py



        return net.state_dict(), sum(epoch_loss) / len(epoch_loss)
        
        # main_fed_cplx.py
        # return net, sum(epoch_loss) / len(epoch_loss)

def download_sampling(w_local, w_glob, prob):
    w_temp = copy.deepcopy(w_local)
    for layer in w_glob.keys():
        if random.random() < prob:
            w_temp[layer] = w_glob[layer]
    return w_temp
