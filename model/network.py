import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pytorch_model_summary import summary
import os
import matplotlib.pyplot as plt

class ConvTasNet(nn.Module):
    def __init__(self, C, N, L, B, H, Sc, P, X, R,
                encoder_activate, mask_activate, causal):
        '''
        C:      Number of source signals
        N:      Numbeer of filters in autoencoder
        L:      Length of filters (in sample)
        B:      Number of channels in bottleneck and the residual path's 1x1-conv blocks
        H:      Number of channels in convolutional blocks
        Sc:     Number of channels in skip-connection path's 1x1-conv blocks
        P:      Kernel size in convolutional blocks
        X:      Number of convolutional blocks in each repeat
        R:      Number of repeats
        causal: Causal or non-causal settting (True or False)
        mask_activate:   activate function of mask estimatin ("sigmoid" or "softmax")
        '''
        super(ConvTasNet, self).__init__()
        
        # Components
        self.encoder = Encoder(L, N, encoder_activate)
        self.separation = Separation(C, N, B, Sc, H, P, X, R, causal, mask_activate)
        self.decoder = Decoder(C, L, N)
        
        # init
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_normal_(p)

    def forward(self, x):
        """
        Args:
            x: [Batchsize, SignalLength] or [SignalLength]
        Returns:
            s_hat: [Batchsize, SignalLength, C]  or [1, SignalLength, C]
        """

        if len(x.shape)==1:
            x = x.unsqueeze(0)

        w = self.encoder(x)
        m = self.separation(w)
        d = w.unsqueeze(3) * m
        s_hat = self.decoder(d)

        s_hat = padding_end(s_hat, x)

        return s_hat


class Encoder(nn.Module):
    def __init__(self, L, N, encoder_activate):
        super().__init__()

        # 50% overlap
        self.conv1d_U = nn.Conv1d(1, N, kernel_size=L, stride=L//2, bias=False)
        if encoder_activate=="ReLU":
            self.activation = nn.ReLU()

    def forward(self, x):
        """
        Args:
            mixture: [Batchsize, Signallength]
        Returns:
            w: [Batchsize, N, Timeframe], where Timeframe = (Signallength-L)/(L/2)+1 = 2*Signallength/L - 1
        """
        x = torch.unsqueeze(x, 1)  # [Batchsize, 1, T]
        w = self.activation(self.conv1d_U(x))  # [Batchsize, N, Timeframe]
        return w


class Decoder(nn.Module):
    def __init__(self, C, L, N):
        super().__init__()

        self.C = C
        self.conv1d_V = nn.ConvTranspose1d(N, 1, kernel_size=L, stride=L//2, bias=False)

    def forward(self, d):
        """
        Args:
            w: [Batchsize, N, Timeframe]
            est_mask: [Batchsize, C, N, Timeframe]
        Returns:
            est_source: [Batchsize, C, T]
        """
        s_hat = []
        for c in range(self.C):
            s_hat.append(self.conv1d_V(d[:,:,:,c]).transpose(1,2))
        
        s_hat = torch.cat(s_hat, dim=2)
        return s_hat


class Separation(nn.Module):
    def __init__(self, C, N, B, Sc, H, P, X, R, causal, mask_activate):
        super().__init__()
        self.network = nn.Sequential()

        self.network.add_module("Normalization", Normalization(N, causal))
        # [Batchsize, N, Timeframe] -> [Batchsize, B, Timeframe]
        self.network.add_module("1x1Conv_input", nn.Conv1d(N, B, 1, bias=False))
        self.network.add_module("TCN", TemporalConvNet(B, Sc, H, P, X, R, causal))
        self.network.add_module("PReLU", nn.PReLU())
        # [Batchsize, B, Timeframe] -> [Batchsize, C*N, Timeframe]
        self.network.add_module("1x1Conv_output", nn.Conv1d(B, C*N, 1, bias=False))
        # [Batchsize, C*N, Timeframe] -> [Batchsize, N, Timeframe, C]
        self.network.add_module("mask", MaskActivation(C, N, mask_activate))

    def forward(self, w):
        """
        Args:
            w: [Batchsize, N, Timeframe]
        returns:
            est_mask: [Batchsize, C, N, Timeframe]
        """
        Batchsize, N, Timeframe = w.size()

        m = self.network(w)

        return m


class TemporalConvNet(nn.Module):
    def __init__(self, B, Sc, H, P, X, R, causal=False):
        super().__init__()
        self.B = B
        self.Sc = Sc
        self.X = X
        self.R = R

        self.network = nn.Sequential()
        for r in range(R):
            for x in range(X):
                dilation = 2**x
                name = "Conv1D, repeat:{}, dilation:{}".format(r,dilation)
                self.network.add_module(name,Conv1DBlock(B, Sc, H, P,
                                        stride=1,
                                        dilation=dilation,
                                        causal=causal))

    def forward(self, mixture_w):
        block_output = mixture_w
        TCN_output = self.network(block_output)[:, self.B:, :]
        return TCN_output


class Conv1DBlock(nn.Module):
    def __init__(self, B, Sc, H, P, stride, dilation, causal=False):
        super().__init__()

        padding = (P - 1) * dilation 
        if not causal:
            padding = padding//2
        
        self.B = B
        self.Sc = Sc

        self.network = nn.Sequential()
        
        # [Batchsize, B, Timeframe] -> [Batchsize, H, Timeframe]
        self.network.add_module("1x1-conv", nn.Conv1d(B, H, 1, bias=False))
        self.network.add_module("PReLU", nn.PReLU())
        self.network.add_module("Normalization", Normalization(H, causal))
        self.network.add_module("D-conv", nn.Conv1d(H, H, P,
                                                    stride=stride, padding=padding,
                                                    dilation=dilation, groups=H,
                                                    bias=False))
        self.network.add_module("PReLU2", nn.PReLU())
        self.network.add_module("Normalization2", Normalization(H, causal))

        # [Batchsize, H, Timeframe] -> [Batchsize, B, Timeframe]
        self.output_conv1x1 = nn.Conv1d(H, B, kernel_size=1, bias=False)
        # [Batchsize, H, Timeframe] -> [Batchsize, Sc, Timeframe]
        self.skip_conv1x1 = nn.Conv1d(H, Sc, kernel_size=1, bias=False)

    def forward(self, x):
        """
        Args:
            x: [Batchsize, B, Timeframe]
            or
            x: [Batchsize, B+Sc, Timeframe]
        Returns:
            [Batchsize, B, Timeframe]
        """
        if x.shape[1] != self.B:
            residual = x[:, self.B:, :]
            skipped = x[:, :self.B, :]
        else:
            residual = x
        convoluted = self.network(residual)
        output = self.output_conv1x1(convoluted) + residual
        skip_connection = self.skip_conv1x1(convoluted)

        if x.shape[1] != self.B:
            skip_connection += skipped

        return torch.cat([output, skip_connection], dim=1)


class Normalization(nn.Module):
    def __init__(self, channel_size, causal):
        super().__init__()
        if causal:
            self.layernorm = CumulativeLayerNormalization(channel_size)
        else:
            self.layernorm = nn.GroupNorm(1, channel_size, eps=1e-8)
    
    def forward(self, x):
        x = self.layernorm(x)
        return x


class MaskActivation(nn.Module):
    def __init__(self, C, N, mask_activate):
        super().__init__()
        self.C = C
        self.N = N
        if mask_activate == 'softmax':
            self.activation = nn.Softmax(dim=3)
        elif mask_activate == 'sigmoid':
            self.activation = nn.Sigmoid()
        else:
            raise ValueError("Unsupported mask activation function")
    
    def forward(self, x):
        Batchsize, _, Timeframe = x.shape
        x = x.view(Batchsize, self.C, self.N, Timeframe)
        x = x.transpose(1,2).transpose(2,3)
        m = self.activation(x)
        return m


class CumulativeLayerNormalization(nn.Module):
    def __init__(self, channel_size):
        super().__init__()
        self.gamma = nn.Parameter(torch.Tensor(1, channel_size, 1))  # [1, N, 1]
        self.beta = nn.Parameter(torch.Tensor(1, channel_size,1 ))  # [1, N, 1]
        self.eps = torch.finfo(torch.float32).eps

        # param initialization
        self.gamma.data.fill_(1)
        self.beta.data.zero_()

    def forward(self, F):
        """
        Args:
            F: [M, N, K], M is batch size, N is channel size, K is length
        Returns:
            cLN_F: [M, N, K]
        """
        mean = F.mean(dim=[1,2], keepdim=True) #[Batchsize, 1, 1]
        var = ((F - mean)**2).mean(dim=[1,2], keepdim=True)
        cLN_F = (F - mean) / (var + self.eps)**0.5 * self.gamma + self.beta
        return cLN_F


# class GlobalLayerNormalization(nn.Module):
#     def __init__(self, channel_size):
#         super().__init__()
#         self.gamma = nn.Parameter(torch.Tensor(1, channel_size, 1))  # [1, N, 1]
#         self.beta = nn.Parameter(torch.Tensor(1, channel_size,1 ))  # [1, N, 1]
#         self.eps = torch.finfo(torch.float32).eps

#         # param initialization
#         self.gamma.data.fill_(1)
#         self.beta.data.zero_()

#     def forward(self, F):
#         """
#         Args:
#             F: [Batchsize, channel_size, Timeframe]
#         Returns:
#             cLN_F: [Batchsize, channel_size, Timeframe]
#         """
#         mean = F.mean(dim=[1,2], keepdim=True) #[Batchsize, 1, 1]
#         var = ((F - mean)**2).mean(dim=[1,2], keepdim=True)
#         gLN_F = (F - mean) / (var + self.eps)**0.5 * self.gamma + self.beta
#         return gLN_F


def padding_end(s_hat, x):
    T_origin = x.size(-1)
    Batchsize, T_conv, C = s_hat.shape
    T_pad = T_origin-T_conv

    padding = torch.zeros([Batchsize, T_pad, C]).to(s_hat.device)
    s_hat =  torch.cat([s_hat,padding],dim=1)

    return s_hat


if __name__ == "__main__":
    C, N, L, B, H, Sc, P, X, R = 2, 128, 40, 128, 256, 128, 3, 7, 2
    encoder_activate, mask, causal = "ReLU", "softmax", False
    ctn = ConvTasNet(C, N, L, B, H, Sc, P, X, R,
                encoder_activate, mask, causal)

    s = torch.randn([5,100001])

    print(ctn(s).shape)