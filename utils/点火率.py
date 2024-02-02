from spikingjelly.activation_based import neuron, layer, functional





def evaluate_FireRate(model, criterion, test_loader):
    """Evaluate classify task model accuracy.

    Returns:
    (loss.sum, acc.avg)
    """
    model.eval()
    gpu = next(model.parameters()).device

    # input = torch.randn(1, 1, 28, 28).to(gpu)
    # flops, params = profile(model, inputs=(input,))
    # print("FLOPs:", flops)
    # print("参数量：", params)

    loss_ = AverageMeter()
    acc_ = AverageMeter()
    firerate_ = AverageMeter()
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(gpu)
            labels = labels.to(gpu)
            # SNN
            functional.reset_net(model)

            # 监视器
            # 监视点火率函数
            def cal_firing_rate(s_seq: torch.Tensor):
                return s_seq.flatten(1).mean(1)

            # 设置监视器
            fr_monitor = monitor.OutputMonitor(model, neuron.IFNode, cal_firing_rate)
            # 监视
            with torch.no_grad():
                functional.reset_net(model)
                fr_monitor.enable()
                outputs = model(inputs)
                _, predicted = torch.max(outputs, 1)
                acc_.update(torch.sum(predicted.eq(labels)).item(), len(labels))
                # print(f'after call fr_monitor.enable(), fr_monitor.records=\n{fr_monitor.records}')
                # print(f'我现在想获取的是第4层的点火率=\n{fr_monitor.records[-1]}')

                # firerate = torch.sum(fr_monitor.records[-1]) #获取最后一层的点火率之和

                # # 获取所有层的点火率
                all_records = torch.cat(fr_monitor.records)
                firerate = torch.sum(all_records)

                # # 最后两层点火率之和
                # last_two_records = torch.cat(fr_monitor.records[-2:])
                # firerate = torch.sum(last_two_records)

                firerate_.update(firerate.item())
                functional.reset_net(model)
                # del fr_monitor
                fr_monitor.remove_hooks()
                # print("最后返回的loss和为",loss_.sum)
            # print("最后返回的点火率之和为", firerate_.sum)

    return firerate_.sum, acc_.avg