#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt
import copy
import numpy as np
from torchvision import datasets, transforms
import torch
import time
import os

from utils.sampling import get_iid, get_noniid, get_mixed_noniid, get_dirichlet_noniid, dataset_stats
from utils.options import args_parser, details
from models.Update import LocalUpdate
from models.Nets import MLP, CNNMnist, CNNCifar, ResidualBlock, ResNetCifar
from models.Fed import FedAvg, fed_hetero
from models.test import test_img
from models.models_complexity import EnhancedCNNCifar, CNNCifarComplexity, init_cplx_dict, SmallCNNCifar, \
    EnhancedCNNFmnist, SmallCNNFmnist
from models.snn_complexity import EnhancedSNNCifar, SNNCifarComplexity, init_cplx_dict_snn, download_hetero_net_snn, SmallSNNCifar, SNN_VGG9
import models.snn_complexity
import random
from collections import Counter
from spikingjelly.activation_based import functional



if __name__ == '__main__':

    start = time.time()

    # seed = 25
    # torch.manual_seed(seed)
    # torch.cuda.manual_seed_all(seed)
    # torch.backends.cudnn.deterministic = True
    # np.random.seed(seed)
    # random.seed(seed)

    seed = 25
    if seed is None:
        seed = random.randint(1,10000)
    print('Random seed: {}'.format(seed))
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
    torch.use_deterministic_algorithms(True)


    # parse args
    args = args_parser()
    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')

    details(args)
    if torch.cuda.is_available():
        print("Using GPU")
        print("Current GPU id:", torch.cuda.current_device())
        print("Current GPU:", torch.cuda.get_device_name(torch.cuda.current_device()))
    else:
        print("Cannot find GPU, return to CPU")

    # load dataset and split users
    if args.dataset == 'mnist':
        trans_mnist = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        dataset_train = datasets.MNIST('./data/mnist/', train=True, download=True, transform=trans_mnist)
        dataset_test = datasets.MNIST('./data/mnist/', train=False, download=True, transform=trans_mnist)
    elif args.dataset == 'fmnist':
        trans_mnist = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        dataset_train = datasets.FashionMNIST('./data/fmnist/', train=True, download=True, transform=trans_mnist)
        dataset_test = datasets.FashionMNIST('./data/fmnist/', train=False, download=True, transform=trans_mnist)
    elif args.dataset == 'cifar':
        trans_cifar = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        dataset_train = datasets.CIFAR10('./data/cifar', train=True, download=True, transform=trans_cifar)
        dataset_test = datasets.CIFAR10('./data/cifar', train=False, download=True, transform=trans_cifar)
    else:
        exit('Error: unrecognized dataset')
    
    if args.iid == 1:
        dict_users = get_iid(dataset_train, args.num_users)
    elif args.iid == 0:
        dict_users = get_noniid(dataset_train, args.num_users)
    elif args.iid == 2:
        dict_users = get_mixed_noniid(dataset_train, args.num_users, args.mixrate)
    elif args.iid == 3:
        dict_users = get_dirichlet_noniid(dataset_train, args.num_users, alpha=args.alpha, sample_num=1000 * args.num_users)
    else:
        exit('Error: unrecognized distribution')
    img_size = dataset_train[0][0].shape
    # print(img_size)
    # visualize data distribution 可视化数据分布
    dataset_stats(dict_users, dataset_train, args)

    
    # build model
    if args.model == 'cnn' and args.dataset == 'cifar':
        # net_glob = CNNCifar(args=args).to(args.device)
        net_glob = EnhancedCNNCifar(args=args).to(args.device)
    elif args.model == 'snn' and args.dataset == 'cifar':
        net_glob = EnhancedSNNCifar(args=args).to(args.device)
        # 为了防止全0输出 使初始化权重全部设置为正值
        # for param in net_glob.parameters():
        #     param.data.abs_()
    elif args.model == 'smallsnn' and args.dataset == 'cifar':
        net_glob = SmallSNNCifar(args=args).to(args.device)

    elif args.model == 'SNN_VGG9' and args.dataset == 'cifar':
        net_glob = SNN_VGG9(args=args).to(args.device)

    elif args.model == 'cnn' and args.dataset == 'fmnist':
        net_glob = EnhancedCNNFmnist(args=args).to(args.device)

    elif args.model == 'cnn' and args.dataset == 'mnist':
        net_glob = CNNMnist(args=args).to(args.device)

    elif args.model == 'smallcnn' and args.dataset == 'cifar':
        net_glob = SmallCNNCifar(args=args).to(args.device)

    elif args.model == 'smallcnn' and args.dataset == 'fmnist':
        net_glob = SmallCNNFmnist(args=args).to(args.device)


    elif args.model == 'resnet' and args.dataset == 'cifar':
        net_glob = ResNetCifar(ResidualBlock, [2, 2, 2]).to(args.device)
    elif args.model == 'mlp':
        net_glob = MLP(input_size=3*1024, hidden_size=500, num_classes=10).to(args.device)
    else:
        exit('Error: unrecognized model')
    print(net_glob)
    net_glob.train()
    # print('前向传播')
    # copy weights
    w_glob = net_glob.state_dict()
    # print('参数：', w_glob)
    # training
    loss_train = []
    acc_test = []
    acc_local = []
    loss_local = []

   
    # if args.all_clients:
    if True:
        print("Aggregation over all clients")
        w_locals = [w_glob for i in range(args.num_users)]
    for iter in range(args.round):
        net_glob.train()
        loss_locals = []
        # if not args.all_clients:
        #     w_locals = []
        m = max(int(args.frac * args.num_users), 1)
        # 选客户端
        # idxs_users = np.random.choice(range(args.num_users), m, replace=False)
        if args.local0andlocal1 == 1:
            idxs_users = np.array([1])
        else:
            idxs_users = np.arange(args.num_users)
        for idx in idxs_users:
            local = LocalUpdate(args=args, dataset=dataset_train, idxs=dict_users[idx], tag=1, idx=idx)
            w, loss = local.train(net=copy.deepcopy(net_glob).to(args.device))
            # if args.all_clients:
            if True:
                w_locals[idx] = copy.deepcopy(w)
            # else:
            #     w_locals.append(copy.deepcopy(w))
            loss_locals.append(copy.deepcopy(loss))


            # 打印客户端准确率
            if args.model == 'smallsnn':
                a = SmallSNNCifar(args=args).to(args.device)
            elif args.model == 'smallcnn' and args.dataset == 'cifar':
                a = SmallCNNCifar(args=args).to(args.device)
            elif args.model == 'smallcnn' and args.dataset == 'fmnist':
                a = SmallCNNFmnist(args=args).to(args.device)
            elif args.model == 'snn' and args.dataset == 'cifar':
                a = EnhancedSNNCifar(args=args).to(args.device)
            elif args.model == 'SNN_VGG9' and args.dataset == 'cifar':
                a = SNN_VGG9(args=args).to(args.device)
            a.load_state_dict(w)
            # print('client {:3d}, loss {:.3f}'.format(idx, loss))
            acc, loss = test_img(a, dataset_test, args)
            print('`````````````````round````````````````````````````````````````````````')
            print("client {:3d} accuracy: {:.2f} loss : {:.3f}".format(idx, acc, loss))
            print('`````````````````round````````````````````````````````````````````````')
            acc_test.append(acc)

        # update global weights


        w_glob = FedAvg(w_locals)

        # copy weight to net_glob
        net_glob.load_state_dict(w_glob)   

        # print loss
        loss_avg = sum(loss_locals) / len(loss_locals)
        print('Round {:3d}, Average loss {:.3f}'.format(iter+1, loss_avg))
        loss_train.append(loss_avg)

        #print accuracy
        acc, loss = test_img(net_glob, dataset_test, args)
        print("Testing accuracy: {:.2f}".format(acc))
        acc_test.append(acc)
        print("-----------{}轮训练结束------------------------------------------".format(iter))
    duration = time.time() - start

    print('\nTotal Run Time: {0:0.4f}'.format(duration))

    # save the data
    np_loss_train = np.array(loss_train)
    np_acc_test = np.array(acc_test)
    np.save('./save/fed_{}_{}_K[{}]_R[{}]_B[{}]_E[{}]_iid[{},{}]_time[{}]_loss[{}].npy'
                .format(args.dataset, args.model, args.num_users, args.round, 
                        args.batchsize, args.epoch, args.iid, args.alpha, duration, loss_avg), np_loss_train)
    np.save('./save/fed_{}_{}_K[{}]_R[{}]_B[{}]_E[{}]_iid[{},{}]_time[{}]_acc[{}].npy'
                .format(args.dataset, args.model, args.num_users, args.round, 
                        args.batchsize, args.epoch, args.iid, args.alpha, duration, acc), np_acc_test)


    # plot loss curve
    plt.figure()
    plt.plot(range(len(loss_train)), loss_train)
    plt.ylabel('train_loss')
    plt.savefig('./save/fed_{}_{}_K[{}]_R[{}]_B[{}]_E[{}]_iid[{},{}]_time[{}]_loss[{}].png'
                .format(args.dataset, args.model, args.num_users, args.round, 
                        args.batchsize, args.epoch, args.iid, args.alpha, duration, loss_avg))
    plt.figure()
    plt.plot(range(len(acc_test)), acc_test)
    plt.title('Test Accuracy vs. Communication Rounds')
    plt.ylabel('Test Accuracy')
    plt.xlabel('Communication Rounds')
    plt.savefig('./save/fed_{}_{}_K[{}]_R[{}]_B[{}]_E[{}]_iid[{},{}]_time[{}]_acc[{}].png'
                .format(args.dataset, args.model, args.num_users, args.round, 
                        args.batchsize, args.epoch, args.iid, args.alpha, duration, acc))


    # testing
    net_glob.eval()
    acc_train, loss_train = test_img(net_glob, dataset_train, args)
    acc_test, loss_test = test_img(net_glob, dataset_test, args)
    print("Training accuracy: {:.2f}".format(acc_train))
    print("Testing accuracy: {:.2f}".format(acc_test))
