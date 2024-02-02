import sys
sys.path.append('.')



import numpy as np
import random
import torch
from torch import nn
import torch.nn.functional as F
from utils.options import args_parser
import copy
from torchinfo import summary
from thop import profile
from spikingjelly.activation_based import neuron, layer, functional


class EnhancedCNNCifar(nn.Module):
    def __init__(self, args):
        super(EnhancedCNNCifar, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(kernel_size=2)

        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(kernel_size=2)

        self.conv5 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn5 = nn.BatchNorm2d(128)
        self.conv6 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn6 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(kernel_size=2)

        self.global_fc1 = nn.Linear(128 * 4 * 4, 128)
        self.global_fc2 = nn.Linear(128, args.num_classes)

    def forward(self, x):
        x = self.bn1(F.relu(self.conv1(x)))
        x = self.bn2(F.relu(self.conv2(x)))
        x = self.pool1(x)

        x = self.bn3(F.relu(self.conv3(x)))
        x = self.bn4(F.relu(self.conv4(x)))
        x = self.pool2(x)

        x = self.bn5(F.relu(self.conv5(x)))
        x = self.bn6(F.relu(self.conv6(x)))
        x = self.pool3(x)

        x = x.view(-1, 128 * 4 * 4)

        x = self.global_fc1(x)
        x = self.global_fc2(x)

        # return F.softmax(x, dim=1)
        return x


class CNNCifarComplexity(nn.Module):
    def __init__(self, args, complexity):
        super(CNNCifarComplexity, self).__init__()

        self.cplx = complexity

        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)

        self.pool = nn.MaxPool2d(kernel_size=2)

        if self.cplx == 1:
            self.fc1 = nn.Linear(32 * 16 * 16, 128) 
            self.fc = nn.Linear(128, args.num_classes)
        
        elif self.cplx == 2:
            self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
            self.bn3 = nn.BatchNorm2d(64)
            self.fc2 = nn.Linear(64 * 8 * 8, 128)
            self.fc = nn.Linear(128, args.num_classes)

        elif self.cplx == 3:
            self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
            self.bn3 = nn.BatchNorm2d(64)
            self.conv4 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
            self.bn4 = nn.BatchNorm2d(64)
            self.fc2 = nn.Linear(64 * 8 * 8, 128)
            self.fc = nn.Linear(128, args.num_classes)

        elif self.cplx == 4:
            self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
            self.bn3 = nn.BatchNorm2d(64)
            self.conv4 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
            self.bn4 = nn.BatchNorm2d(64)
            self.conv5 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
            self.bn5 = nn.BatchNorm2d(128 )
            self.fc3 = nn.Linear(128 * 4 * 4, 128)
            self.fc = nn.Linear(128, args.num_classes)
        
        elif self.cplx == 5:
            self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
            self.bn3 = nn.BatchNorm2d(64)
            self.conv4 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
            self.bn4 = nn.BatchNorm2d(64)
            self.conv5 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
            self.bn5 = nn.BatchNorm2d(128)
            self.conv6 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
            self.bn6 = nn.BatchNorm2d(128)  
            self.global_fc1 = nn.Linear(128 * 4 * 4, 128)
            self.global_fc2 = nn.Linear(128, args.num_classes)
        
    def forward(self, x):
        x = self.bn1(F.relu(self.conv1(x)))
        x = self.bn2(F.relu(self.conv2(x)))
        x = self.pool(x)

        if self.cplx == 1:
            x = x.view(-1, 32 * 16 * 16)
            x = self.fc1(x)
            x = self.fc(x)
            return x

        x = self.bn3(F.relu(self.conv3(x)))

        if self.cplx == 2:
            x = self.pool(x)
            x = x.view(-1, 64 * 8 * 8)
            x = self.fc2(x)
            x = self.fc(x)
            return x

        x = self.bn4(F.relu(self.conv4(x)))
        x = self.pool(x)

        if self.cplx == 3:
            x = x.view(-1, 64 * 8 * 8)
            x = self.fc2(x)
            x = self.fc(x)
            return x

        x = self.bn5(F.relu(self.conv5(x)))

        if self.cplx == 4:
            x = self.pool(x)
            x = x.view(-1, 128 * 4 * 4)
            x = self.fc3(x)
            x = self.fc(x)
            return x
        
        x = self.bn6(F.relu(self.conv6(x)))
        x = self.pool(x)

        if self.cplx == 5:
            x = x.view(-1, 128 * 4 * 4)
            x = self.global_fc1(x)
            x = self.global_fc2(x)
            return x


def init_cplx_dict(args):
    idx_cplx_dict={}                     # {idxs_user: complexity  -->  0: 5, 1: 1, 2: 4, ...}
    for i in range(args.num_users):
        idx_cplx_dict[i] = int( np.random.choice([1,2,3,4,5],1) )
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

# def generate_cplx_net(args, global_model, user_idx, idx_cplx_dict, device='cuda'):   # download
#     cplx = idx_cplx_dict[user_idx]
#     local_model = CNNCifarComplexity(args, cplx).to(device)
#     local_update = copy.deepcopy(local_model.state_dict())
#     if cplx == 5:
#         for name in local_update.keys():
#             local_update[name] = global_model.state_dict()[name]
#         local_model.load_state_dict(local_update)
#     else:
#         for name in local_update.keys():	
#             if 'fc' in name:
#                 continue
#             else:
#                 local_update[name] = global_model.state_dict()[name]
#         local_model.load_state_dict(local_update)
#     return local_model

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



if __name__=="__main__":
    args = args_parser()
    input = torch.randn(1, 3, 32, 32).to('cuda')
    for i in [1, 2, 3, 4, 5]:
        print(i, '--------------------------------------------------------------------------------')
        model = SNNCifarComplexity(args, i).to('cuda')
        summary(model, (1, 3, 32, 32))

        flops, params = profile(model, inputs=(input, ))
        print('flops:========================================================')

        print(flops)
        print('params:========================================================')
        print(params)
