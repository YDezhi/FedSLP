#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @python: 3.6

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from spikingjelly.activation_based import monitor, neuron, functional, layer


def test_img(net_g, datatest, args):
    firerate = args.firerate
    net_g.eval()
    # testing
    test_loss = 0
    correct = 0
    data_loader = DataLoader(datatest, batch_size=args.test_bs)

    l = len(data_loader)
    for idx, (data, target) in enumerate(data_loader):
        if args.gpu != -1:
            # if args.model == 'snn' or args.model == 'smallsnn' or args.model == 'snnandann' or args.model == 'smallsnnandann' or args.model == 'snnbntt' or args.model == 'SNN_VGG9':
            if args.model in ['snn', 'smallsnn', 'snnandann', 'smallsnnandann', 'SNN_VGG9', 'snnvgg9', 'snnbntt']:
                data, target = data.cuda(), target.cuda()
                target_onehot = F.one_hot(target, 10).float()
            else:
                data, target = data.cuda(), target.cuda()

        # 12.4 RuntimeError: Input type (torch.cuda.FloatTensor) and weight type (torch.FloatTensor) should be the same 报错更改
        net_g = net_g.cuda()
        log_probs = net_g(data)
        # sum up batch loss
        # print(target)
        test_loss += F.cross_entropy(log_probs, target_onehot, reduction='sum').item()
        # get the index of the max log-probability
        if args.model != ('snn' and 'smallsnn' and 'snnandann' and 'smallsnnandann' and 'snnbntt'):
            y_pred = log_probs.data.max(1, keepdim=True)[1]
            # print('pred:', y_pred)
            correct += y_pred.eq(target.data.view_as(y_pred)).long().cpu().sum()
        else:
            #print(log_probs.argmax(1))
            correct += (log_probs.argmax(1) == target).float().sum().item()
        # 重置网络
        if args.model == 'snn' or 'smallsnn' or 'snnandann' or 'smallsnnandann':
            functional.reset_net(net_g)


    test_loss /= len(data_loader.dataset)
    accuracy = 100.00 * correct / len(data_loader.dataset)
    if args.verbose:
        print('\nTest set: Average loss: {:.4f} \nAccuracy: {}/{} ({:.2f}%)\n'.format(
            test_loss, correct, len(data_loader.dataset), accuracy))

    # 监视点火率
    if firerate == 1:
        def cal_firing_rate(s_seq: torch.Tensor):
            # s_seq.shape = [T, N, *]
            return s_seq.flatten(1).mean(1)

        # 设置监视器
        fr_monitor = monitor.OutputMonitor(net_g, neuron.LIFNode, cal_firing_rate)
        # 监视
        with torch.no_grad():
            # functional.reset_net(net)
            # fr_monitor.disable()
            # net(images)
            # functional.reset_net(net)
            # print(f'after call fr_monitor.disable(), fr_monitor.records=\n{fr_monitor.records}')
            fr_monitor.enable()
            a = net_g(data)
            # print(a)
            print(f'after call fr_monitor.enable(), fr_monitor.records=\n{fr_monitor.records}')
            functional.reset_net(net_g)
            del fr_monitor
    return accuracy, test_loss
    #
