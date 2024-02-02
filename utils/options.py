#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import argparse

def args_parser():
    parser = argparse.ArgumentParser()
    # federated arguments
    parser.add_argument('--round', type=int, default=100, help="rounds of training") # 300
    parser.add_argument('--num_users', type=int, default=10, help="number of users: K")
    parser.add_argument('--frac', type=float, default=1, help="the fraction of clients: C")
    parser.add_argument('--epoch', type=int, default=5, help="the number of local epochs: E")
    parser.add_argument('--batchsize', type=int, default=32, help="local batch size: B") # 64
    parser.add_argument('--test_bs', type=int, default=32, help="test batch size")
    parser.add_argument('--lr', type=float, default=0.1, help="learning rate") # 0.001
    parser.add_argument('--momentum', type=float, default=0.5, help="SGD momentum (default: 0.5)")
    parser.add_argument('--split', type=str, default='user', help="train-test split type, user or sample")
    parser.add_argument('--lr_interval', default='0.33 0.66', type=str,
                        help='intervals at which to reduce lr, expressed as %%age of total epochs')
    parser.add_argument('--lr_reduce', default=10, type=int, help='reduction factor for learning rate')

    # model arguments
    parser.add_argument('--model', type=str, default='snnvgg9', help='model name')
    parser.add_argument('--optimizer', type=str, default='sgd', help='Optimizer, sgd or adam') # sgd
    parser.add_argument('--kernel_num', type=int, default=9, help='number of each kind of kernel')
    parser.add_argument('--kernel_sizes', type=str, default='3,4,5', help='comma-separated kernel size to use for convolution')
    parser.add_argument('--norm', type=str, default='batch_norm', help="batch_norm, layer_norm, or None")
    parser.add_argument('--num_filters', type=int, default=32, help="number of filters for conv nets")
    parser.add_argument('--max_pool', type=str, default='True', help="Whether use max pooling rather than strided convolutions")

    # snn arguments
    parser.add_argument('--T', type=int, default=20, help='步长')
    parser.add_argument('--threshold', type=float, default=1., help='电压阈值')
    parser.add_argument('--threshold2', type=float, default=1., help='电压阈值2')
    parser.add_argument('--threshold3', type=float, default=1., help='电压阈值3')

    parser.add_argument('--local0andlocal1', type=int, default=0, help='只用前两个客户端')
    parser.add_argument('--tau', type=float, default=1.5, help='时间常数')

    parser.add_argument('--firerate', type=int, default=0, help='1=输出点火率，0=不输出点火率')

    # other arguments
    parser.add_argument('--dataset', type=str, default='cifar', help="name of dataset")
    parser.add_argument('--iid', type=int, default=3, help='whether i.i.d or not')
    parser.add_argument('--mixrate', type=float, default=0.95, help='mixed non-iid rate')
    parser.add_argument('--alpha', type=float, default=1., help='dirichlet non-iid rate')
    parser.add_argument('--num_classes', type=int, default=10, help="number of classes")
    parser.add_argument('--num_channels', type=int, default=3, help="number of channels of imges")
    parser.add_argument('--gpu', type=int, default=0, help="GPU ID, -1 for CPU")
    parser.add_argument('--stopping_rounds', type=int, default=10, help='rounds of early stopping')
    parser.add_argument('--verbose', action='store_true', help='verbose print')
    parser.add_argument('--seed', type=int, default=1, help='random seed (default: 1)')
    parser.add_argument('--all_clients', action='store_true', help='aggregation over all clients')
    parser.add_argument('--up_droprate', type=float, default=1, help='dropout rate')
    args = parser.parse_args()
    return args

def details(args):
    print('\nExperimental details:')
    print(f'Dataset            : {args.dataset}')
    if args.iid == 1:
        print('IID')
    elif args.iid == 0:
        print('Non-IID')
    else:
        print('Partly-IID')
    if args.model == 'snn':
        print(f'电压阈值             : {args.threshold}')
        print(f'电压阈值2            : {args.threshold2}')
        print(f'电压阈值3            : {args.threshold3}')
    print(f'时间步T             : {args.T}')
    print(f'alpha              : {args.alpha}')
    print(f'Upload Dropout rate: {args.up_droprate}')
    print(f'Model              : {args.model}')
    print(f'lr                 : {args.lr}')
    print(f'Optimizer          : {args.optimizer}')
    print(f'Global Rounds      : {args.round}')
    print(f'Local [E] Epoch    : {args.epoch}')
    print(f'Local [B] Batchsize: {args.batchsize}')
    print(f'Local [C] Fraction : {args.frac}')
    print(f'# Clients          : {args.num_users}\n')

    
    return
