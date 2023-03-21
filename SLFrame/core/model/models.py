#!/usr/bin/python3

"""models.py Contains an implementation of the LeNet5 model

For the ID2223 Scalable Machine Learning course at KTH Royal Institute of
Technology"""

__author__ = "Xenia Ioannidou and Bas Straathof"

import torch
import torch.nn as nn
import torch.nn.functional as F


class LeNetComplete(nn.Module):
    """CNN based on the classical LeNet architecture, but with ReLU instead of
    tanh activation functions and max pooling instead of subsampling."""

    def __init__(self):
        super(LeNetComplete, self).__init__()

        # First block - convolutional
        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))

        # Second block - convolutional
        self.block2 = nn.Sequential(
            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.flatten = nn.Flatten()
        # Third block - fully connected
        self.block3 = nn.Sequential(
            nn.Linear(in_features=256, out_features=120),
            nn.ReLU(),
            nn.Linear(in_features=120, out_features=84),
            nn.ReLU(),
            nn.Linear(in_features=84, out_features=10))

    def forward(self, x):
        """Define forward pass of CNN

        Args:
            x: Input Tensor

        Returns:
          x: Output Tensor
        """
        # Apply first convolutional block to input tensor
        x = self.block1(x)
        # Apply second convolutional block to input tensor
        x = self.block2(x)

        # Flatten output
        x = x.view(x.size(0), -1)

        # Apply first fully-connected block to input tensor
        x = self.block3(x)

        return x


class LeNetClientNetwork(nn.Module):
    """CNN following the architecture of:
    https://medium.com/ml2vec/intro-to-pytorch-with-image-classification-on-a- \
            fashion-clothes-dataset-e589682df0c5

    The ClientNetwork is used for Split Learning and implements the CNN
    until the first convolutional layer."""

    def __init__(self):
        super(LeNetClientNetwork, self).__init__()

        # First block - convolutional
        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))

        # Second block - convolutional

    def forward(self, x):
        """Defines forward pass of CNN until the split layer, which is the first
        convolutional layer

        Args:
            x: Input Tensor

        Returns:
          x: Output Tensor
        """
        # Apply first convolutional block to input tensor
        x1 = self.block1(x)

        return x1


class LeNetServerNetwork(nn.Module):
    """CNN following the architecture of:
    https://medium.com/ml2vec/intro-to-pytorch-with-image-classification-on-a- \
            fashion-clothes-dataset-e589682df0c5

    The ServerNetwork is used for Split Learning and implements the CNN
    from the split layer until the last."""

    def __init__(self):
        super(LeNetServerNetwork, self).__init__()

        # Second block - convolutional
        self.block2 = nn.Sequential(
            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))

        # Third block - fully connected
        self.block3 = nn.Sequential(
            nn.Linear(in_features=256, out_features=120),
            nn.ReLU(),
            nn.Linear(in_features=120, out_features=84),
            nn.ReLU(),
            nn.Linear(in_features=84, out_features=10))

    def forward(self, x):
        """Defines forward pass of CNN from the split layer until the last

        Args:
            x: Input Tensor

        Returns:
          x: Output Tensor
        """
        # Apply second convolutional block to input tensor
        # x = self.block2(x)
        # Flatten output
        #
        x = self.block2(x)
        x = x.view(-1, 4*4*16)
        # Apply fully-connected block to input tensor
        x = self.block3(x)

        return x


class adult_LR(nn.Module):
    def __init__(self):
        super(adult_LR, self).__init__()
        # First block - convolutional
        self.block1 = nn.Sequential(
            nn.Linear(14, 10),

        )
        self.block2 = nn.Sequential(
            nn.Linear(10, 7),

        )
        self.block3 = nn.Sequential(
            nn.Linear(7, 2),

        )

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        return x


class adult_LR_server(nn.Module):
    def __init__(self):
        super(adult_LR_server, self).__init__()
        # First block - convolutional

        self.block2 = nn.Sequential(
            nn.Linear(10, 7),

        )
        self.block3 = nn.Sequential(
            nn.Linear(7, 2),

        )

    def forward(self, x):
        # x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        # x = torch.sigmoid(x)
        return x


class adult_LR_client(nn.Module):
    def __init__(self):
        super(adult_LR_client, self).__init__()
        # First block - convolutional
        self.block1 = nn.Sequential(
            nn.Linear(14, 10),

        )

    def forward(self, x):
        x = self.block1(x)

        return x


class german_LR_client(nn.Module):
    def __init__(self):
        super(german_LR_client, self).__init__()
        # First block - convolutional
        self.block1 = nn.Sequential(
            nn.Linear(24, 14),
        )

    def forward(self, x):
        x = self.block1(x)

        return x


class german_LR_server(nn.Module):
    def __init__(self):
        super(german_LR_server, self).__init__()
        # First block - convolutional

        self.block2 = nn.Sequential(
            nn.Linear(14, 7),

        )
        self.block3 = nn.Sequential(
            nn.Linear(7, 2),

        )

    def forward(self, x):
        # x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        return x


class adult_LR_client1(nn.Module):
    def __init__(self):
        super(adult_LR_client1, self).__init__()
        # First block - convolutional
        self.block1 = nn.Sequential(
            nn.Linear(6, 4),

        )

    def forward(self, x):
        x = self.block1(x)
        # x = torch.sigmoid(x)
        return x


class adult_LR_client2(nn.Module):
    def __init__(self):
        super(adult_LR_client2, self).__init__()
        # First block - convolutional
        self.block1 = nn.Sequential(
            nn.Linear(4, 3),

        )

    def forward(self, x):
        x = self.block1(x)
        # x = torch.sigmoid(x)
        return x


# class adult_LR_client3(nn.Module):
#     def __init__(self):
#         super(adult_LR_client3, self).__init__()
#         # First block - convolutional
#         self.block1 = nn.Sequential(
#             nn.Linear(8, 6),
#
#         )
#
#     def forward(self, x):
        x = self.block1(x)
        # x = torch.sigmoid(x)
        return x
