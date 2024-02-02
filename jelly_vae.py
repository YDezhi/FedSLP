# -*- coding:utf-8 -*-
"""
@author: QgZhan
@contact: zhanqg@foxmail.com
@file: jelly_vae.py
@time：2023/3/22 12:11
"""


import torch
import torch.nn as nn
from spikingjelly.activation_based import neuron, functional, surrogate, layer
from torch.autograd import Variable
import torch.nn.init as init
from models.jelly.jelly_prior import PriorBernoulliSTBP
from models.jelly.jelly_posterior import PosteriorBernoulliSTBP


class JellyVAE(nn.Module):
    def __init__(self, opt, device):
        """
        初始化Jelly版本的VAE网络
        """
        super(JellyVAE, self).__init__()
        self.decoder_type = opt.decoder_type
        self.sample_type = opt.sample_type
        self.use_cupy = opt.use_cupy
        self.output_size = opt.output_size
        self.latent_size = opt.latent_size
        self.T = opt.T
        self.k = opt.k
        self.device = device
        self.output_neuron_threshold = opt.output_neuron_threshold
        if opt.decoder_type == 'transpose':
            self.decoder_input_size = opt.decoder_input_size

        # 构造特征提取器feature
        self.feature = None

        # 构造编码器encoder
        self.encoder = None

        # 构造解码器
        if self.decoder_type == "transpose":
            # 构造构造decoder输入
            self.decoder_input = nn.Sequential(
                layer.Linear(opt.latent_size, opt.feature_config[-1][1] * opt.decoder_input_size * opt.decoder_input_size),
                # layer.BatchNorm2d(256 * 4),
                neuron.LIFNode(tau=opt.tau, v_threshold=opt.v_threshold, v_reset=opt.v_reset,
                               surrogate_function=opt.surrogate_function)
            )

            # 构造decoder
            modules = []
            for i, config in enumerate(opt.decoder_config):
                modules.append(layer.ConvTranspose2d(in_channels=config[0],
                                                     out_channels=config[1],
                                                     kernel_size=config[2],
                                                     stride=config[3],
                                                     padding=config[4],
                                                     output_padding=config[5],
                                                     bias=True)
                               )
                if i < len(opt.decoder_config)-1:
                    modules.append(layer.BatchNorm2d(config[1]))
                    modules.append(neuron.LIFNode(tau=opt.tau, v_threshold=opt.v_threshold, v_reset=opt.v_reset,
                                                  surrogate_function=opt.surrogate_function))
                else:
                    modules.append(neuron.LIFNode(tau=opt.tau, v_threshold=opt.output_neuron_threshold, v_reset=opt.v_reset,
                                                  surrogate_function=opt.surrogate_function, store_v_seq=True))

            self.decoder = nn.Sequential(*modules)

        elif self.decoder_type == 'linear':
            # 构造decoder
            modules = []
            for i, config in enumerate(opt.decoder_config):
                modules.append(layer.Linear(config[0], config[1]))
                if i < len(opt.decoder_config) - 1:
                    modules.append(neuron.LIFNode(tau=opt.tau, v_threshold=opt.v_threshold, v_reset=opt.v_reset,
                                                  surrogate_function=opt.surrogate_function))
                else:
                    modules.append(neuron.LIFNode(tau=opt.tau, v_threshold=opt.output_neuron_threshold, v_reset=opt.v_reset,
                                                  surrogate_function=opt.surrogate_function, store_v_seq=True))
            self.decoder = nn.Sequential(*modules)

        # 构造重建器
        if opt.output_size[0] == 2:
            self.reconstruction = nn.Sigmoid()
        elif opt.output_size[0] == 3:
            self.reconstruction = nn.Tanh()

        # 构造隐变量生成器
        if self.sample_type == 'network':
            self.prior = PriorBernoulliSTBP(opt)
            self.posterior = PosteriorBernoulliSTBP(opt)

        self._set_model()

    def forward(self, x: torch.Tensor):
        # Compress to latent space
        # 压缩输入到潜在空间
        pass

        # Compute latent vector and sample z
        # 计算隐变量，进行采样
        pass

        # Decode to event frame given a latent vector
        # 解码事件帧给定的潜在向量
        pass

        # 图像重建
        pass

    def generate_sample_z(self, latent_x):
        """
        生成采样的sample_z
        :param latent_x: 隐变量
        :return: sampled_z-通过隐变量采样得到的z, param_1, param_2与隐变量采样方法有关。
        """
        if self.sample_type == 'gaussian':
            mu = latent_x[..., : self.latent_size]  # [T, N, latent_size*2] -> [T, N, latent_size]
            log_var = latent_x[..., self.latent_size:]  # [T, N, latent_size*2] -> [T, N, latent_size]
            sampled_z = self._sample_latent(mu, log_var)  # -> [T, N, latent_size]
            param_1 = mu  # [T, N, latent_size]
            param_2 = log_var  # [T, N, latent_size]
        elif self.sample_type == 'network':
            sampled_z, q_z = self.posterior(
                latent_x)  # [T, N, latent_size] -> [T, N, latent_size], [T, N, latent_size, k]
            p_z = self.prior(sampled_z)  # -> [T, N, latent_size, k]
            param_1 = q_z  # [T, N, latent_size, k]
            param_2 = p_z  # [T, N, latent_size, k]
        elif self.sample_type == 'poison':
            rate_mu = latent_x.mean(dim=0)  # [T, N, latent_size] -> [N, latent_size]
            print(rate_mu.shape, rate_mu.max(), rate_mu.min(), rate_mu.mean(), rate_mu.std())
            print(rate_mu[0, :3])
            sampled_z, _ = self.poison_reparameterize(rate_mu)  # [T, N, latent_size],
            param_1 = rate_mu    #  [N, latent_size]
            param_2 = sampled_z  #  [T, N, latent_size]
        else:
            raise RuntimeError(f"sample_type should be 'poison', 'gaussian' or 'network', "
                               f"but your sample_type is {self.sample_type}")

        return sampled_z, param_1, param_2

    def decode_sample_z(self, sampled_z):
        if self.decoder_type == "transpose":
            decoder_z = self.decoder_input(sampled_z)  # [T, N, latent_size] -> [T, N, opt.feature_config[-1][1] * 4]
            decoder_z = decoder_z.view([sampled_z.shape[0], sampled_z.shape[1], -1, self.decoder_input_size,
                                        self.decoder_input_size])  # [T, N, opt.feature_config[-1][1] * 4] -> [T, N, opt.feature_config[-1][1], 2, 2]
            decoder_z = self.decoder(decoder_z)  # [T, N, C, W, H]
        elif self.decoder_type == 'linear':
            decoder_z = self.decoder(sampled_z)  # [T, N, C*W*H]
        else:
            raise RuntimeError(f"decoder_type should be 'transpose' or 'linear', "
                               f"but your decoder_type is {self.decoder_type}")
        return decoder_z
        
    def sample(self, batch_size=64):
        latent_x = torch.rand((self.T, batch_size, self.latent_size)) > 0.5
        latent_x = latent_x.float().to(self.device)

        # Compute latent vector and sample z
        # 计算隐变量，进行采样
        sampled_z, param_1, param_2 = self.generate_sample_z(latent_x)

        # Decode to event frame given a latent vector
        # 解码事件帧给定的潜在向量
        decoder_z = self.decode_sample_z(sampled_z)

        # 图像重建
        if self.output_neuron_threshold < torch.inf:  # 输出为传统0、1值
            output = decoder_z.view(self.T, batch_size, self.output_size[0], self.output_size[1], self.output_size[2])
        else:    # 输出膜电位作为连续值
            mem = self.decoder._modules[str(int(len(self.decoder._modules)-1))].v_seq
            output = self.reconstruction(mem[-1, ...])
            output = output.view(batch_size, self.output_size[0], self.output_size[1], self.output_size[2])

        return output, param_1, param_2, sampled_z

    def _set_model(self):
        self._initialize_weights()

        functional.set_step_mode(self, step_mode='m')

        if self.use_cupy:
            functional.set_backend(self, backend='cupy')

    def _initialize_weights(self):
        for m in self.modules():
            # 判断是否属于Conv2d
            if isinstance(m, (layer.Conv1d, layer.Conv2d)):
                init.kaiming_normal_(m.weight.data)
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

    def _sample_latent(self, mu, log_var):
        """
        通过重参数技巧对隐变量对应的分布采样
        :param mu: 均值
        :param log_var: 方差
        :return:
        """
        std = torch.exp(0.5 * log_var)
        eps = Variable(std.data.new(std.size()).normal_())

        return mu + eps * std

    def poison_reparameterize(self, mu):
        lif_node = neuron.LIFNode(decay_input=False, v_threshold=mu)  # [N, latent_size]
        x = torch.rand([self.T, mu.shape[0], mu.shape[1]]).to(mu.device)  # [T, N, latent_size]
        sample_z = lif_node(x)

        # print(x[:, 0, :3])  # [T, 3]
        # print(sample_z[:, 0, :3])
        #
        # print(x.shape, x.max(), x.min(), x.mean(), x.std())
        # print(sample_z.shape, sample_z.max(), sample_z.min(), sample_z.mean(), sample_z.std())
        return sample_z, sample_z

    # def poison_reparameterize(self, mu):
    #     batch_size, self.latent_dim = mu.shape
    #     random_indices = []
    #     mu = mu.unsqueeze(-1).repeat(1, 1, self.k).unsqueeze(-1).repeat(1, 1, 1, 2)  # (N, latent_dim, K, 2), 其中K表示采样K次，2表示采样结果为0和1的取值概率
    #     mu[..., 0] = 1 - mu[..., 1]  # [..., 0] 表示采样结果为0的概率，[..., 1] 表示采样结果为1的概率
    #
    #     # input z_t_minus again to calculate tdBN
    #     sampled_z = None
    #     full_sampled_z = None
    #
    #     for t in range(self.T):
    #         # sampling
    #         full_sampled_z_t = self._gumbel_softmax_sampling(mu)   # (N, latent_dim, K, 2)\
    #
    #         # 将sampled_z_t转化为0/1序列
    #         add = (1-torch.round(full_sampled_z_t[..., 1])) * full_sampled_z_t[..., 0]
    #         full_sampled_z_t = torch.round(full_sampled_z_t[..., 1]) * full_sampled_z_t[..., 1] + add     # (N, latent_dim, K)
    #
    #         # random_index
    #         random_index = torch.randint(0, self.k, (batch_size * self.latent_dim,)) + \
    #                        torch.arange(start=0, end=batch_size * self.latent_dim * self.k, step=self.k)
    #         random_indices.append(random_index)  # shape: [batch_size*latent_dim,]
    #
    #         # sampling
    #         sampled_z_t = full_sampled_z_t.view(batch_size * self.latent_dim * self.k)[random_index]  # (B*C,)
    #         sampled_z_t = sampled_z_t.view(1, batch_size, self.latent_dim)  # (B,C,1)
    #
    #         if t == 0:
    #             sampled_z = sampled_z_t
    #             full_sampled_z = full_sampled_z_t.unsqueeze(0)
    #         else:
    #             sampled_z = torch.cat([sampled_z, sampled_z_t], dim=0)
    #             full_sampled_z = torch.cat([full_sampled_z, full_sampled_z_t.unsqueeze(0)], dim=0)
    #
    #     return sampled_z, full_sampled_z

    def _gumbel_softmax_sampling(self, mu, gumbel_mu=0, gumbel_beta=1, tau=0.1):
        """
        mu : (N, latent_dim, K) tensor. Assume we need to sample a N*latent_dim*K tensor, each row is an independent r.v.
        """
        shape_mu = mu.shape
        y = torch.rand(shape_mu).to(mu.device) + 1e-25  # ensure all y is positive.
        g = self._inverse_gumbel_cdf(y, gumbel_mu, gumbel_beta).to(mu.device)
        x = torch.log(mu + 1e-7) + g  # samples follow Gumbel distribution.
        # using softmax to generate one_hot vector:
        x = x / tau
        x = nn.functional.softmax(x, dim=-1)  # now, the x approximates a one_hot vector.
        return x

    def _inverse_gumbel_cdf(self, y, gumbel_mu, gumbel_beta):
        return gumbel_mu - gumbel_beta * torch.log(-torch.log(y))


