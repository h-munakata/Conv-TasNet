import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pytorch_model_summary import summary
import os
import matplotlib.pyplot as plt

class ConvTasNet(nn.Module):
    def __init__(self, config):
        super(ConvTasNet, self).__init__()
        self.N = config['network']['N']
        self.L = config['network']['L']
        self.B = config['network']['B']
        self.Sc = config['network']['Sc']
        self.H = config['network']['H']
        self.P = config['network']['P']
        self.X = config['network']['X']
        self.R = config['network']['R']
        self.C = config['network']['C']
        self.causal = config['network']['causal']
        self.af = config['network']['af']
        self.device = torch.device(config['gpu'])
        # Components
        self.encoder = Encoder(self.L, self.N)
        self.separation = Separation(self.N, self.B, self.Sc, self.H, self.P,self.X, 
                                        self.R, self.C, causal=self.causal, af=self.af)
        self.decoder = Decoder(self.N, self.L)
        # init
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_normal_(p)

    def forward(self, mixture):
        """
        Args:
            mixture: [Batchsize, T]
        Returns:
            est_source: [Batchsize, T, C]
        """
        mixture_w = self.encoder(mixture)
        est_mask = self.separation(mixture_w)
        X,Y,Z = mixture_w.shape
        # est_mask = torch.ones(X*Y*Z*2).reshape([X,self.C,Y,Z])
        est_s = self.decoder(mixture_w, est_mask)

        T_origin = mixture.size(-1)
        Batchsize,T_conv,C = est_s.shape
        padding = torch.zeros([Batchsize,T_origin-T_conv,C]).to(est_s.device)
        est_s =  torch.cat([est_s,padding],dim=1)

        return est_s




class Encoder(nn.Module):
    def __init__(self, L, N):
        super().__init__()

        # 50% overlap
        self.conv1d_U = nn.Conv1d(1, N, kernel_size=L, stride=L//2, bias=False)

    def forward(self, mixture):
        """
        Args:
            mixture: [Batchsize, Signallength]
        Returns:
            w: [Batchsize, N, Timeframe], where Timeframe = (Signallength-L)/(L/2)+1 = 2*Signallength/L-1
        """
        mixture = torch.unsqueeze(mixture, 1)  # [Batchsize, 1, T]
        w = F.relu(self.conv1d_U(mixture))  # [Batchsize, N, Timeframe]
        return w


class Decoder(nn.Module):
    def __init__(self, N, L):
        super().__init__()

        self.conv1d_V = nn.ConvTranspose1d(N, 1, kernel_size=L, stride=L//2, bias=False)

    def forward(self, w, est_mask):
        """
        Args:
            w: [Batchsize, N, Timeframe]
            est_mask: [Batchsize, C, N, Timeframe]
        Returns:
            est_source: [Batchsize, C, T]
        """
        # D = W * M
        # [Batchsize, N, Timeframe] * C
        C = est_mask.shape[1]
        d = torch.unsqueeze(w, 1) * est_mask  # [Batchsize, C, N, Timeframe]
        
        # S = DV
        # [Batchsize, Timeframe, L] * C -> [Batchsize, T]*C
        s = [self.conv1d_V(d[:,c,:,:]).transpose(1,2) for c in range(C)]
        s = torch.cat(s,dim=2)
        return s


class Separation(nn.Module):
    def __init__(self, N, B, Sc, H, P, X, R, C, causal=False, af='relu'):
        super().__init__()
        # Hyper-parameter
        self.C = C
        self.af = af

        self.network = nn.Sequential()

        self.network.add_module("LayerNorm", GlobalLayerNormalization(N))
        # [Batchsize, N, Timeframe] -> [Batchsize, B, Timeframe]
        self.network.add_module("1x1Conv_bottleneck", nn.Conv1d(N, B, 1, bias=False))
        self.network.add_module("Temporal Convolutional Network", TemporalConvNet(B, Sc, H, P, X, R, causal))
        self.network.add_module("PReLU", nn.PReLU())
        # [Batchsize, B, Timeframe] -> [Batchsize, C*N, Timeframe]
        self.network.add_module("1x1Conv_mask_est", nn.Conv1d(B, C*N, 1, bias=False))

        if af == 'softmax':
            self.network.add_module("Sigmoid", nn.Softmax(dim=1))
        elif af == 'relu':
            self.network.add_module("Sigmoid", nn.ReLU())
        else:
            raise ValueError("Unsupported mask activation function")

    def forward(self, w):
        """
        Args:
            w: [Batchsize, N, Timeframe]
        returns:
            est_mask: [Batchsize, C, N, Timeframe]
        """
        Batchsize, N, Timeframe = w.size()

        est_mask = self.network(w)
        est_mask = est_mask.view(Batchsize, self.C, N, Timeframe)

        return est_mask


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
        padding = (P - 1) * dilation if causal else (P - 1) * dilation // 2
        self.B = B
        self.Sc = Sc

        self.network = nn.Sequential()
        # [Batchsize, B, Timeframe] -> [Batchsize, H, Timeframe]
        self.network.add_module("1x1-conv", nn.Conv1d(B, H, 1, bias=False))
        self.network.add_module("PReLU", nn.PReLU())
        self.network.add_module("Normalization", GlobalLayerNormalization(H))
        self.network.add_module("D-conv", nn.Conv1d(H, H, P,
                                                    stride=stride, padding=padding,
                                                    dilation=dilation, groups=H,
                                                    bias=False))
        self.network.add_module("PReLU", nn.PReLU())
        self.network.add_module("Normalization", GlobalLayerNormalization(H))

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



# class CumulativeLayerNormalization(nn.Module):
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
#             F: [M, N, K], M is batch size, N is channel size, K is length
#         Returns:
#             cLN_F: [M, N, K]
#         """
#         mean = torch.mean(F, dim=1, keepdim=True)  # [M, 1, K]
#         var = torch.var(F, dim=1, keepdim=True, unbiased=False)  # [M, 1, K]
#         cLN_F = (F - mean) / (var + self.eps)**5 * self.gamma + self.beta
#         return cLN_F


class GlobalLayerNormalization(nn.Module):
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
            F: [Batchsize, channel_size, Timeframe]
        Returns:
            cLN_F: [Batchsize, channel_size, Timeframe]
        """
        mean = F.mean(dim=[1,2], keepdim=True) #[Batchsize, 1, 1]
        var = ((F - mean)**2).mean(dim=[1,2], keepdim=True)
        gLN_F = (F - mean) / (var + self.eps)**0.5 * self.gamma + self.beta
        return gLN_F


if __name__ == "__main__":
    torch.manual_seed(123)
    Batchsize, N, L, Signallength = 2, 16, 24, 100
    Timeframe = 2*Signallength//L-1
    B, Sc, H, P, X, R, C, norm_type, causal = 2, 2, 3, 3, 3, 2, 2, "gLN", False

    mixture = torch.rand(Batchsize, Signallength)

    enc = Encoder(L,N)
    # [Batchsize, N, Timeframe]
    enc_vec = enc(mixture)
    print(enc_vec.shape)

    sep = Separation(N, B, H, P, X, R, C)
    sep_vec = sep(enc_vec)
    # [Batchsize, C, N, Timeframe]
    print(sep_vec.shape)