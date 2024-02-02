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
        x = x.unsqueeze(0).repeat(self.T, 1, 1, 1, 1)

        x = self.sn1(self.bn1(self.conv1(x)))
        x = self.sn2(self.bn2(self.conv2(x)))
        x = self.pool1(x)

        x = self.sn3(self.bn3(self.conv3(x)))
        x = self.sn4(self.bn4(self.conv4(x)))
        x = self.pool2(x)

        # 更改sn5sn6查看数据
        x = self.conv5(x)
        x = self.bn5(x)
        x = self.sn5(x)
        x = self.conv6(x)
        x = self.bn6(x)
        x = self.sn6(x)
        x = self.conv7(x)
        x = self.bn7(x)
        x = self.sn7(x)
        x = self.pool3(x)
        x = self.flat(x)


        x = self.global_fc1(x)
        x = self.sn8(x)
        x = self.global_fc2(x)
        x = self.sn9(x)

        x = x.mean(0)

        return x