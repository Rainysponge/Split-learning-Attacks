import math

import torch
import torch.nn as nn
import torch.nn.functional as F

SCALING_PARAMS = {  # (width_coefficient, depth_coefficient, resolution, dropout_rate)
    'efficientnet-b0': (1.0, 1.0, 224, 0.2),
    'efficientnet-b1': (1.0, 1.1, 240, 0.2),
    'efficientnet-b2': (1.1, 1.2, 260, 0.3),
    'efficientnet-b3': (1.2, 1.4, 300, 0.3),
    'efficientnet-b4': (1.4, 1.8, 380, 0.4),
    'efficientnet-b5': (1.6, 2.2, 456, 0.4),
    'efficientnet-b6': (1.8, 2.6, 528, 0.5),
    'efficientnet-b7': (2.0, 3.1, 600, 0.5)}


# --------------------
# Model components
# --------------------

class Swish(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x * torch.sigmoid(x)


class Squeeze(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        x = x.squeeze()
        if x.dim() == 1:
            x = x.unsqueeze(0)  # keep a batch dim
        return x


class DropConnect(nn.Dropout3d):
    # implemented here via dropout3d (cf torch docs): randomly zero out entire channels
    #   (a channel is a 3D feature map, e.g. the j-th channel of the i-th sample in the batched input is
    #   a 3D tensor input[i,j]) where layer input is (N,C,D,H,W).
    #   Here -- if we unsqueeze input to (1,N,C,H,W) dropout3d effectively zeros out datapoints, which matches the
    #   implementation by the authors multiplying the inputs by a binary tensor of shape (batch_size,1,1,1)
    def forward(self, x):
        return F.dropout3d(x.unsqueeze(0), self.p, self.training, self.inplace).squeeze(0)


class PaddedConv2d(nn.Conv2d):
    def forward(self, x):
        # `same` padding (cf https://www.tensorflow.org/api_docs/python/tf/nn/convolution)
        print(x.shape)
        h_in, w_in = x.shape[2:]
        h_out, w_out = math.ceil(h_in / self.stride[0]), math.ceil(w_in / self.stride[1])
        padding = (
        math.ceil(max((h_out - 1) * self.stride[0] - h_in + self.dilation[0] * (self.kernel_size[0] - 1) + 1, 0) / 2),
        math.ceil(max((w_out - 1) * self.stride[1] - h_in + self.dilation[1] * (self.kernel_size[1] - 1) + 1, 0) / 2))

        if padding[0] > 0 or padding[1] > 0:
            x = F.pad(x, [padding[1], padding[1], padding[0], padding[0]])

        return super().forward(x)


class SELayer(nn.Sequential):
    """ Squeeze and excitation layer cf Squeeze-and-Excitation Networks https://arxiv.org/abs/1709.01507 figure 2/3"""

    def __init__(self, n_channels, se_reduce_channels):
        super().__init__(nn.AdaptiveAvgPool2d(1),
                         nn.Conv2d(n_channels, se_reduce_channels, kernel_size=1),
                         Swish(),
                         nn.Conv2d(se_reduce_channels, n_channels, kernel_size=1),
                         nn.Sigmoid())

    def forward(self, x):
        return x * super().forward(x)


class MBConvBlock(nn.Sequential):
    """ mobile inverted residual bottleneck cf MnasNet https://arxiv.org/abs/1807.11626 figure 7b """

    def __init__(self, in_channels, out_channels, kernel_size, stride, expand_ratio, se_ratio, drop_connect_rate):
        expand_channels = int(in_channels * expand_ratio)
        se_reduce_channels = max(1, int(in_channels * se_ratio))

        modules = []
        # expansion conv
        if expand_ratio != 1:
            modules += [nn.Conv2d(in_channels, expand_channels, kernel_size=1, bias=False),
                        nn.BatchNorm2d(expand_channels),
                        Swish()]
        modules += [
            # depthwise conv
            PaddedConv2d(expand_channels, expand_channels, kernel_size, stride, groups=expand_channels, bias=False),
            nn.BatchNorm2d(expand_channels),
            Swish(),
            # squeeze and excitation
            SELayer(expand_channels, se_reduce_channels),
            # projection conv
            nn.Conv2d(expand_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels)]

        # drop connect -- only apply drop connect if skip (skip connection (in_channles==out_channels and stride=1))
        if in_channels == out_channels and stride == 1:
            modules += [DropConnect(drop_connect_rate)]

        super().__init__(*modules)

    def forward(self, x):
        out = super().forward(x)
        if out.shape == x.shape:  # skip connection (in_channles==out_channels and stride=1)
            return out + x
        return out


class MBConvBlockRepeat(nn.Sequential):
    """ repeats MBConvBlock """

    def __init__(self, n_repeats, in_channels, out_channels, kernel_size, stride, expand_ratio, se_ratio,
                 drop_connect_rate):
        self.n_repeats = n_repeats
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.expand_ratio = expand_ratio
        self.se_ratio = se_ratio
        self.drop_connect_rate = drop_connect_rate

        modules = []
        for i in range(n_repeats):
            modules += [MBConvBlock(in_channels, out_channels, kernel_size, stride, expand_ratio, se_ratio,
                                    drop_connect_rate * i / n_repeats)]
            in_channels = out_channels
            stride = 1
        super().__init__(*modules)


# --------------------
# Model
# --------------------

class EfficientNetB0(nn.Module):
    """ efficientnet b0 cf https://arxiv.org/abs/1905.11946 table 1 """

    def __init__(self, n_classes, dropout_rate=0.2, drop_connect_rate=0.2, bn_eps=1e-3, bn_momentum=0.01):
        super().__init__()

        self.stem = nn.Sequential(
            PaddedConv2d(3, 32, kernel_size=3, stride=2, bias=False),
            nn.BatchNorm2d(32),
            Swish())

        self.blocks = nn.Sequential(
            # blocks init (n_repeat, in_channels, out_channels, kernel_size, stride, expand_ratio, se_ratio)
            MBConvBlockRepeat(1, 32, 16, 3, 1, 1, 0.25, drop_connect_rate),
            MBConvBlockRepeat(2, 16, 24, 3, 2, 6, 0.25, drop_connect_rate),
            MBConvBlockRepeat(2, 24, 40, 5, 2, 6, 0.25, drop_connect_rate),
            MBConvBlockRepeat(3, 40, 80, 3, 2, 6, 0.25, drop_connect_rate),
            MBConvBlockRepeat(3, 80, 112, 5, 1, 6, 0.25, drop_connect_rate),
            MBConvBlockRepeat(4, 112, 192, 5, 2, 6, 0.25, drop_connect_rate),
            MBConvBlockRepeat(1, 192, 320, 3, 1, 6, 0.25, drop_connect_rate))

        self.head = nn.Sequential(
            nn.Conv2d(320, 1280, kernel_size=1, bias=False),
            nn.BatchNorm2d(1280),
            Swish(),
            nn.AdaptiveAvgPool2d(1),
            Squeeze(),
            nn.Dropout(dropout_rate),
            nn.Linear(1280, n_classes))

        # initialize
        self.bn_eps = bn_eps
        self.bn_momentum = bn_momentum
        self.reset_parameters()

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eps = self.bn_eps
                m.momentum = self.bn_momentum
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='conv2d')
                if m.bias is not None: nn.init.constant_(m.bias, 0)
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, a=math.sqrt(5), mode='fan_in', nonlinearity='linear')
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        return self.head(self.blocks(self.stem(x)))


class EfficientNetB0_Client(nn.Module):
    """ efficientnet b0 cf https://arxiv.org/abs/1905.11946 table 1 """

    def __init__(self, n_classes=1, dropout_rate=0.2, drop_connect_rate=0.2, bn_eps=1e-3, bn_momentum=0.01):
        super().__init__()

        self.stem = nn.Sequential(
            PaddedConv2d(3, 32, kernel_size=3, stride=2, bias=False),
            nn.BatchNorm2d(32),
            Swish())

        self.blocks = nn.Sequential(
            # blocks init (n_repeat, in_channels, out_channels, kernel_size, stride, expand_ratio, se_ratio)
            MBConvBlockRepeat(1, 32, 16, 3, 1, 1, 0.25, drop_connect_rate),
            MBConvBlockRepeat(2, 16, 24, 3, 2, 6, 0.25, drop_connect_rate),
        )

        # initialize
        self.bn_eps = bn_eps
        self.bn_momentum = bn_momentum
        self.reset_parameters()

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eps = self.bn_eps
                m.momentum = self.bn_momentum
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='conv2d')
                if m.bias is not None: nn.init.constant_(m.bias, 0)
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, a=math.sqrt(5), mode='fan_in', nonlinearity='linear')
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        return self.blocks(self.stem(x))


class EfficientNetB0_Server(nn.Module):
    """ efficientnet b0 cf https://arxiv.org/abs/1905.11946 table 1 """

    def __init__(self, n_classes=2, dropout_rate=0.2, drop_connect_rate=0.2, bn_eps=1e-3, bn_momentum=0.01):
        super().__init__()

        self.blocks = nn.Sequential(
            # blocks init (n_repeat, in_channels, out_channels, kernel_size, stride, expand_ratio, se_ratio)

            MBConvBlockRepeat(2, 24, 40, 5, 2, 6, 0.25, drop_connect_rate),
            MBConvBlockRepeat(3, 40, 80, 3, 2, 6, 0.25, drop_connect_rate),
            MBConvBlockRepeat(3, 80, 112, 5, 1, 6, 0.25, drop_connect_rate),
            MBConvBlockRepeat(4, 112, 192, 5, 2, 6, 0.25, drop_connect_rate),
            MBConvBlockRepeat(1, 192, 320, 3, 1, 6, 0.25, drop_connect_rate))

        self.head = nn.Sequential(
            nn.Conv2d(320, 1280, kernel_size=1, bias=False),
            nn.BatchNorm2d(1280),
            Swish(),
            nn.AdaptiveAvgPool2d(1),
            Squeeze(),
            nn.Dropout(dropout_rate),
            nn.Linear(1280, n_classes))

        # initialize
        self.bn_eps = bn_eps
        self.bn_momentum = bn_momentum
        self.reset_parameters()

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eps = self.bn_eps
                m.momentum = self.bn_momentum
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='conv2d')
                if m.bias is not None: nn.init.constant_(m.bias, 0)
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, a=math.sqrt(5), mode='fan_in', nonlinearity='linear')
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        return self.head(self.blocks(x))
