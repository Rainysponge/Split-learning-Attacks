import torch
import torch.nn as nn
import torch.nn.functional as F


class CNN_OriginalFedAvg(torch.nn.Module):
    """The CNN model used in the original FedAvg paper:
    "Communication-Efficient Learning of Deep Networks from Decentralized Data"
    https://arxiv.org/abs/1602.05629.

    The number of parameters when `only_digits=True` is (1,663,370), which matches
    what is reported in the paper.
    When `only_digits=True`, the summary of returned model is

    Model:
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #
    =================================================================
    reshape (Reshape)            (None, 28, 28, 1)         0
    _________________________________________________________________
    conv2d (Conv2D)              (None, 28, 28, 32)        832
    _________________________________________________________________
    max_pooling2d (MaxPooling2D) (None, 14, 14, 32)        0
    _________________________________________________________________
    conv2d_1 (Conv2D)            (None, 14, 14, 64)        51264
    _________________________________________________________________
    max_pooling2d_1 (MaxPooling2 (None, 7, 7, 64)          0
    _________________________________________________________________
    flatten (Flatten)            (None, 3136)              0
    _________________________________________________________________
    dense (Dense)                (None, 512)               1606144
    _________________________________________________________________
    dense_1 (Dense)              (None, 10)                5130
    =================================================================
    Total params: 1,663,370
    Trainable params: 1,663,370
    Non-trainable params: 0

    Args:
      only_digits: If True, uses a final layer with 10 outputs, for use with the
        digits only MNIST dataset (http://yann.lecun.com/exdb/mnist/).
        If False, uses 62 outputs for Federated Extended MNIST (FEMNIST)
        EMNIST: Extending MNIST to handwritten letters: https://arxiv.org/abs/1702.05373.
    Returns:
      A `torch.nn.Module`.
    """

    def __init__(self, only_digits=True):
        super(CNN_OriginalFedAvg, self).__init__()
        self.only_digits = only_digits
        self.conv2d_1 = torch.nn.Conv2d(1, 32, kernel_size=5, padding=2)
        self.max_pooling = nn.MaxPool2d(2, stride=2)
        self.conv2d_2 = torch.nn.Conv2d(32, 64, kernel_size=5, padding=2)
        self.flatten = nn.Flatten()
        self.linear_1 = nn.Linear(3136, 512)
        self.linear_2 = nn.Linear(512, 10 if only_digits else 62)
        self.relu = nn.ReLU()
        # self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = torch.unsqueeze(x, 1)
        x = self.conv2d_1(x)
        x = self.relu(x)
        x = self.max_pooling(x)
        x = self.conv2d_2(x)
        x = self.relu(x)
        x = self.max_pooling(x)
        x = self.flatten(x)
        x = self.relu(self.linear_1(x))
        x = self.linear_2(x)
        # x = self.softmax(self.linear_2(x))
        return x


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # 输入shape 3*32*32
        self.conv1 = nn.Conv2d(3, 64, 3, padding=1)  # 64*32*32
        self.conv2 = nn.Conv2d(64, 64, 3, padding=1)  # 64*32*32
        self.pool1 = nn.MaxPool2d(2, 2)  # 64*16*16
        self.bn1 = nn.BatchNorm2d(64)  # 64*16*16
        self.relu1 = nn.ReLU()  # 64*16*16

        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)  # 128*16*16
        self.conv4 = nn.Conv2d(128, 128, 3, padding=1)  # 128*16*16
        self.pool2 = nn.MaxPool2d(2, 2, padding=1)  # 128*9*9
        self.bn2 = nn.BatchNorm2d(128)  # 128*9*9
        self.relu2 = nn.ReLU()  # 128*9*9

        self.conv5 = nn.Conv2d(128, 128, 3, padding=1)  # 128*9*9
        self.conv6 = nn.Conv2d(128, 128, 3, padding=1)  # 128*9*9
        self.conv7 = nn.Conv2d(128, 128, 1, padding=1)  # 128*11*11
        self.pool3 = nn.MaxPool2d(2, 2, padding=1)  # 128*6*6
        self.bn3 = nn.BatchNorm2d(128)  # 128*6*6
        self.relu3 = nn.ReLU()  # 128*6*6

        self.conv8 = nn.Conv2d(128, 256, 3, padding=1)  # 256*6*6
        self.conv9 = nn.Conv2d(256, 256, 3, padding=1)  # 256*6*6
        self.conv10 = nn.Conv2d(256, 256, 1, padding=1)  # 256*8*8
        self.pool4 = nn.MaxPool2d(2, 2, padding=1)  # 256*5*5
        self.bn4 = nn.BatchNorm2d(256)  # 256*5*5
        self.relu4 = nn.ReLU()  # 256*5*5

        self.conv11 = nn.Conv2d(256, 512, 3, padding=1)  # 512*5*5
        self.conv12 = nn.Conv2d(512, 512, 3, padding=1)  # 512*5*5
        self.conv13 = nn.Conv2d(512, 512, 1, padding=1)  # 512*7*7
        self.pool5 = nn.MaxPool2d(2, 2, padding=1)  # 512*4*4
        self.bn5 = nn.BatchNorm2d(512)  # 512*4*4
        self.relu5 = nn.ReLU()  # 512*4*4

        self.fc14 = nn.Linear(512 * 4 * 4, 1024)  # 1*1024
        self.drop1 = nn.Dropout2d()  # 1*1024
        self.fc15 = nn.Linear(1024, 1024)  # 1*1024
        self.drop2 = nn.Dropout2d()  # 1*1024
        self.fc16 = nn.Linear(1024, 10)  # 1*10

    def forward(self, x):
        # x = x.to(device)  # 自加
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.pool1(x)
        x = self.bn1(x)
        x = self.relu1(x)

        x = self.conv3(x)
        x = self.conv4(x)
        x = self.pool2(x)
        x = self.bn2(x)
        x = self.relu2(x)

        x = self.conv5(x)
        x = self.conv6(x)
        x = self.conv7(x)
        x = self.pool3(x)
        x = self.bn3(x)
        x = self.relu3(x)

        x = self.conv8(x)
        x = self.conv9(x)
        x = self.conv10(x)
        x = self.pool4(x)
        x = self.bn4(x)
        x = self.relu4(x)

        x = self.conv11(x)
        x = self.conv12(x)
        x = self.conv13(x)
        x = self.pool5(x)
        x = self.bn5(x)
        x = self.relu5(x)
        # print(" x shape ",x.size())
        x = x.view(-1, 512 * 4 * 4)
        x = F.relu(self.fc14(x))
        x = self.drop1(x)
        x = F.relu(self.fc15(x))
        x = self.drop2(x)
        x = self.fc16(x)

        return x


class cifar10_client(nn.Module):
    def __init__(self):
        super(cifar10_client, self).__init__()
        # 输入shape 3*32*32
        self.conv1 = nn.Conv2d(3, 64, 3, padding=1)  # 64*32*32
        self.conv2 = nn.Conv2d(64, 64, 3, padding=1)  # 64*32*32
        self.pool1 = nn.MaxPool2d(2, 2)  # 64*16*16
        self.bn1 = nn.BatchNorm2d(64)  # 64*16*16
        self.relu1 = nn.ReLU()  # 64*16*16

    def forward(self, x):
        # x = x.to(device)  # 自加
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.pool1(x)
        x = self.bn1(x)
        x = self.relu1(x)

        return x


class cifar10_server(nn.Module):
    def __init__(self):
        super(cifar10_server, self).__init__()
        # 输入shape 3*32*32

        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)  # 128*16*16
        self.conv4 = nn.Conv2d(128, 128, 3, padding=1)  # 128*16*16
        self.pool2 = nn.MaxPool2d(2, 2, padding=1)  # 128*9*9
        self.bn2 = nn.BatchNorm2d(128)  # 128*9*9
        self.relu2 = nn.ReLU()  # 128*9*9

        self.conv5 = nn.Conv2d(128, 128, 3, padding=1)  # 128*9*9
        self.conv6 = nn.Conv2d(128, 128, 3, padding=1)  # 128*9*9
        self.conv7 = nn.Conv2d(128, 128, 1, padding=1)  # 128*11*11
        self.pool3 = nn.MaxPool2d(2, 2, padding=1)  # 128*6*6
        self.bn3 = nn.BatchNorm2d(128)  # 128*6*6
        self.relu3 = nn.ReLU()  # 128*6*6

        self.conv8 = nn.Conv2d(128, 256, 3, padding=1)  # 256*6*6
        self.conv9 = nn.Conv2d(256, 256, 3, padding=1)  # 256*6*6
        self.conv10 = nn.Conv2d(256, 256, 1, padding=1)  # 256*8*8
        self.pool4 = nn.MaxPool2d(2, 2, padding=1)  # 256*5*5
        self.bn4 = nn.BatchNorm2d(256)  # 256*5*5
        self.relu4 = nn.ReLU()  # 256*5*5

        self.conv11 = nn.Conv2d(256, 512, 3, padding=1)  # 512*5*5
        self.conv12 = nn.Conv2d(512, 512, 3, padding=1)  # 512*5*5
        self.conv13 = nn.Conv2d(512, 512, 1, padding=1)  # 512*7*7
        self.pool5 = nn.MaxPool2d(2, 2, padding=1)  # 512*4*4
        self.bn5 = nn.BatchNorm2d(512)  # 512*4*4
        self.relu5 = nn.ReLU()  # 512*4*4

        self.fc14 = nn.Linear(512 * 4 * 4, 1024)  # 1*1024
        self.drop1 = nn.Dropout2d()  # 1*1024
        self.fc15 = nn.Linear(1024, 1024)  # 1*1024
        self.drop2 = nn.Dropout2d()  # 1*1024
        self.fc16 = nn.Linear(1024, 10)  # 1*10

    def forward(self, x):
        # x = x.to(device)  # 自加

        x = self.conv3(x)
        x = self.conv4(x)
        x = self.pool2(x)
        x = self.bn2(x)
        x = self.relu2(x)

        x = self.conv5(x)
        x = self.conv6(x)
        x = self.conv7(x)
        x = self.pool3(x)
        x = self.bn3(x)
        x = self.relu3(x)

        x = self.conv8(x)
        x = self.conv9(x)
        x = self.conv10(x)
        x = self.pool4(x)
        x = self.bn4(x)
        x = self.relu4(x)

        x = self.conv11(x)
        x = self.conv12(x)
        x = self.conv13(x)
        x = self.pool5(x)
        x = self.bn5(x)
        x = self.relu5(x)
        # print(" x shape ",x.size())
        x = x.view(-1, 512 * 4 * 4)
        x = F.relu(self.fc14(x))
        x = self.drop1(x)
        x = F.relu(self.fc15(x))
        x = self.drop2(x)
        x = self.fc16(x)

        return x

