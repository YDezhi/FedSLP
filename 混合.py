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
from models.snn_complexity import EnhancedSNNCifar, SNNCifarComplexity, init_cplx_dict_snn, download_hetero_net_snn, \
    SmallSNNCifar, EnhancedSNNFmnist, SmallSNNFmnist
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
        seed = random.randint(1, 10000)
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
        trans_cifar = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
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
        dict_users = get_dirichlet_noniid(dataset_train, args.num_users, alpha=args.alpha,
                                          sample_num=1000 * args.num_users)
    else:
        exit('Error: unrecognized distribution')
    img_size = dataset_train[0][0].shape
    # print(img_size)
    # visualize data distribution 可视化数据分布
    dataset_stats(dict_users, dataset_train, args)

    # build model

    if args.model == 'snnandann' and args.dataset == 'fmnist':
        net_glob_cnn = EnhancedCNNFmnist(args=args).to(args.device)
        net_glob_snn = EnhancedSNNFmnist(args=args).to(args.device)
    elif args.model == 'smallsnnandann' and args.dataset == 'fmnist':
        net_glob_cnn = SmallCNNFmnist(args=args).to(args.device)
        net_glob_snn = SmallSNNFmnist(args=args).to(args.device)
    else:
        exit('Error: unrecognized model')

    print("net_glob_snn:", net_glob_snn)
    print("net_glob_cnn:", net_glob_cnn)
    net_glob_cnn.train()
    net_glob_snn.train()
    # print('前向传播')
    # copy weights


    # 聚合模型为ann
    w_glob = net_glob_cnn.state_dict()


    # print('参数：', w_glob)
    # training
    loss_train = []
    acc_test = []

    acc_cnn_test = []
    acc_snn_test = []

    acc_local = []
    loss_local = []

    # 模型分配
    cnn_client_indices = np.array([0, 1, 2, 3, 4])
    snn_client_indices = np.array([5, 6, 7, 8, 9])



    # if args.all_clients:
    if True:
        print("Aggregation over all clients")
        w_locals = [w_glob for i in range(args.num_users)]
    for iter in range(args.round):
        net_glob_cnn.train()
        net_glob_snn.train()
        loss_locals = []
        # if not args.all_clients:
        #     w_locals = []
        m = max(int(args.frac * args.num_users), 1)
        # 选客户端
        # idxs_users = np.random.choice(range(args.num_users), m, replace=False)

        # 选择分配不同客户端的索引表



        for idx in cnn_client_indices:
            local = LocalUpdate(args=args, dataset=dataset_train, idxs=dict_users[idx])
            w, loss = local.train(net=copy.deepcopy(net_glob_cnn).to(args.device))
            # if args.all_clients:
            if True:
                w_locals[idx] = copy.deepcopy(w)
            # else:
            #     w_locals.append(copy.deepcopy(w))
            loss_locals.append(copy.deepcopy(loss))

            # 打印客户端准确率
            if args.model == 'snnandann':
                a = EnhancedCNNFmnist(args=args).to(args.device)
            elif args.model == 'smallsnnandann':
                a = SmallCNNFmnist(args=args).to(args.device)
            a.load_state_dict(w)
            # print('client {:3d}, loss {:.3f}'.format(idx, loss))
            acc, loss = test_img(a, dataset_test, args)
            print("client {:3d} accuracy: {:.2f} loss : {:.3f}".format(idx, acc, loss))
            # acc_test.append(acc)


        for idx in snn_client_indices:
            local = LocalUpdate(args=args, dataset=dataset_train, idxs=dict_users[idx], tag=1)
            w, loss = local.train(net=copy.deepcopy(net_glob_snn).to(args.device))
            # if args.all_clients:
            if True:
                w_locals[idx] = copy.deepcopy(w)
            # else:
            #     w_locals.append(copy.deepcopy(w))
            loss_locals.append(copy.deepcopy(loss))

            # 打印客户端准确率
            if args.model == 'snnandann':
                a = EnhancedSNNFmnist(args=args).to(args.device)
            elif args.model == 'smallsnnandann':
                a = SmallSNNFmnist(args=args).to(args.device)
            a.load_state_dict(w)
            # print('client {:3d}, loss {:.3f}'.format(idx, loss))
            acc, loss = test_img(a, dataset_test, args)
            print("client {:3d} accuracy: {:.2f} loss : {:.3f}".format(idx, acc, loss))
            # acc_test.append(acc)

        # update global weights

        w_glob = FedAvg(w_locals)

        # copy weight to net_glob
        net_glob_cnn.load_state_dict(w_glob)
        net_glob_snn.load_state_dict(w_glob)

        # print loss
        loss_avg = sum(loss_locals) / len(loss_locals)
        print('Round {:3d}, Average loss {:.3f}'.format(iter + 1, loss_avg))
        loss_train.append(loss_avg)

        # print accuracy
        acc_cnn, loss_cnn = test_img(net_glob_cnn, dataset_test, args)
        acc_snn, loss_snn = test_img(net_glob_snn, dataset_test, args)
        print("cnn Testing accuracy: {:.2f}".format(acc_cnn))
        print("snn Testing accuracy: {:.2f}".format(acc_snn))
        acc_cnn_test.append(acc_cnn)
        acc_snn_test.append(acc_snn)
        print("-----------{}轮训练结束------------------------------------------".format(iter))
    duration = time.time() - start

    print('\nTotal Run Time: {0:0.4f}'.format(duration))

    # save the data
    np_loss_train = np.array(loss_train)
    np_cnn_acc_test = np.array(acc_cnn_test)
    np_snn_acc_test = np.array(acc_snn_test)


    np.save('./save/fed_loss_{}_{}_K[{}]_R[{}]_B[{}]_E[{}]_iid[{},{}]_time[{}]_loss[{}].npy'
            .format(args.dataset, args.model, args.num_users, args.round,
                    args.batchsize, args.epoch, args.iid, args.alpha, duration, loss_avg), np_loss_train)
    # 存ann的acc
    np.save('./save/fed_ann_{}_{}_K[{}]_R[{}]_B[{}]_E[{}]_iid[{},{}]_time[{}]_acc[{}].npy'
            .format(args.dataset, args.model, args.num_users, args.round,
                    args.batchsize, args.epoch, args.iid, args.alpha, duration, acc_cnn), np_cnn_acc_test)
    # 存snn的acc
    np.save('./save/fed_snn_{}_{}_K[{}]_R[{}]_B[{}]_E[{}]_iid[{},{}]_time[{}]_acc[{}].npy'
            .format(args.dataset, args.model, args.num_users, args.round,
                    args.batchsize, args.epoch, args.iid, args.alpha, duration, acc_snn), np_snn_acc_test)



    # plot loss curve
    plt.figure()
    plt.plot(range(len(loss_train)), loss_train)
    plt.ylabel('train_loss')
    plt.savefig('./save/fed_{}_{}_K[{}]_R[{}]_B[{}]_E[{}]_iid[{},{}]_time[{}]_loss[{}].png'
                .format(args.dataset, args.model, args.num_users, args.round,
                        args.batchsize, args.epoch, args.iid, args.alpha, duration, loss_avg))

    # 画ann的acc
    plt.figure()
    plt.plot(range(len(acc_cnn_test)), acc_cnn_test)
    plt.title('Test ann Accuracy vs. Communication Rounds')
    plt.ylabel('Test ann Accuracy')
    plt.xlabel('Communication Rounds')
    plt.savefig('./save/fed_ann_{}_{}_K[{}]_R[{}]_B[{}]_E[{}]_iid[{},{}]_time[{}]_acc[{}].png'
                .format(args.dataset, args.model, args.num_users, args.round,
                        args.batchsize, args.epoch, args.iid, args.alpha, duration, acc_cnn))

    # 画snn的acc
    plt.figure()
    plt.plot(range(len(acc_snn_test)), acc_snn_test)
    plt.title('Test snn Accuracy vs. Communication Rounds')
    plt.ylabel('Test snn Accuracy')
    plt.xlabel('Communication Rounds')
    plt.savefig('./save/fed_snn_{}_{}_K[{}]_R[{}]_B[{}]_E[{}]_iid[{},{}]_time[{}]_acc[{}].png'
                .format(args.dataset, args.model, args.num_users, args.round,
                        args.batchsize, args.epoch, args.iid, args.alpha, duration, acc_snn))

    # testing
    net_glob_cnn.eval()
    net_glob_snn.eval()

    acc_cnn_train, loss_train = test_img(net_glob_cnn, dataset_train, args)
    acc_cnn_test, loss_test = test_img(net_glob_cnn, dataset_test, args)

    acc_snn_train, loss_train = test_img(net_glob_snn, dataset_train, args)
    acc_snn_test, loss_test = test_img(net_glob_snn, dataset_test, args)

    print("ann Training accuracy: {:.2f}".format(acc_cnn_train))
    print("ann Testing accuracy: {:.2f}".format(acc_cnn_test))

    print("snn Training accuracy: {:.2f}".format(acc_snn_train))
    print("snn Testing accuracy: {:.2f}".format(acc_snn_test))

