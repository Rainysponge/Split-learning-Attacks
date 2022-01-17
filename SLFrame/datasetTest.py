import numpy as np
import pandas as pd
import torch

from torchvision.datasets import MNIST

if __name__ == "__main__":
    # mnist_dataobj = MNIST("./data/mnist", True, transform=None, target_transform=None, download=False)
    #
    # data = mnist_dataobj.data
    # target = np.array(mnist_dataobj.targets)
    #
    # print(type(mnist_dataobj))
    # print(type(data))
    # print((data.size()))
    # print(type(target))
    # print(target.size)
    data = np.loadtxt("./data/german/german.data-numeric")
    n, l = data.shape
    for j in range(l - 1):
        meanVal = np.mean(data[:, j])
        stdVal = np.std(data[:, j])
        data[:, j] = (data[:, j] - meanVal) / stdVal
    np.random.shuffle(data)

    # print(type(test_lab))
