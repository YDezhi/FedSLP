def _initialize_weights(self):
    for m in self.modules():
    # 判断是否属于Conv2d
        if isinstance(m, (layer.Conv1d, layer.Conv2d)):
            torch.nn.init.kaiming_normal_(m.weight.data)
        # 判断是否有偏置
        if m.bias is not None:
            torch.nn.init.zeros_(m.bias.data)
        elif isinstance(m, layer.Linear):
            torch.nn.init.kaiming_normal_(m.weight.data)
        if m.bias is not None:
            torch.nn.init.zeros_(m.bias.data)
        elif isinstance(m, (layer.BatchNorm1d, layer.BatchNorm2d)):
            m.weight.data.fill_(1)
            torch.nn.init.zeros_(m.bias.data)