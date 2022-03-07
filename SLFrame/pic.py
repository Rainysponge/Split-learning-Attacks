import numpy as np
import pandas as pd
import torch
import seaborn
from matplotlib import pyplot as plt
import seaborn as sns
import re
from queue import Queue
from torchvision import transforms
from torchvision.datasets import FashionMNIST

from Parse.YAML.yamlParse import yamlParse


def pic():
    with open("D:/Split-learning-Attacks/SABuf/Split-learning-Attacks/SLFrame/log.txt", "r") as f:
        content = f.read()

    p = re.compile("phase=validation acc=(.*?) loss")
    matchObj = re.findall(p, content)
    acc = [float(matchObj[i]) for i in range(int(len(matchObj)))]
    print(acc)
    # acc = [(float(matchObj[i * 3]) + float(matchObj[3 * i + 1]) + float(matchObj[3 * i + 2])) / 3 for i in
    #        range(int(len(matchObj)) // 3)]
    # acc.insert(0, 0.59453125)
    # acc = [float(matchObj[i]) for i in range(len(matchObj))]
    x = [i for i in range(len(acc))]
    data = np.array([x, acc])

    df = pd.DataFrame(data.T, columns=["turn", "acc"])
    df["turn"] = df["turn"].astype(int)
    df["acc"] = df["acc"].astype(float)
    print(data)
    print(df.dtypes)
    sns.lineplot(x="turn", y="acc", data=df)
    plt.show()


def client_pic():
    with open("D:/Split-learning-Attacks/SABuf/Split-learning-Attacks/SLFrame/log.txt", "r") as f:
        content = f.read()
    string = "(.*?) - phase=train acc=(.*?) loss"
    pid = []
    p = re.compile("INFO - (.*?) - phase=validation acc=(.*?) loss")
    matchObj = re.findall(p, content)
    acc = [float(matchObj[i][1]) for i in range(int(len(matchObj)))]
    hue = [matchObj[i][0] for i in range(int(len(matchObj)))]
    # x = [ for i in range(len(acc) // 3)]
    time_point = []
    client_set = list(set(hue))
    counter = dict()
    for i in hue:
        if i not in counter:
            counter[i] = 0
        else:
            counter[i] += 1
        time_point.append(counter[i])
    data = np.array([time_point, hue, acc])
    df = pd.DataFrame(data.T, columns=["turn", "client", "acc"])
    # print(data)

    df["turn"] = df["turn"].astype(int)
    df["acc"] = df["acc"].astype(float)
    # df["client"] = df["client"].astype(str)
    # print(df.dtypes)
    print(df["client"])
    # df["client"] = df["client"].astype(int)
    sns.pointplot(x="turn", y="acc", hue="client", data=df)
    plt.show()


if __name__ == "__main__":
    # q = Queue(maxsize=0)
    # aa = (1, 2)
    # q.put(aa)
    # q.put(2)
    # print(id(aa) == id(q.get()))
    # x = 1
    # print(id(x))
    # x = x + 1
    # print(id(x))
    # x = np.array([1,2,3,4,5,6,5, 2,1,4,5,4,3,1])
    # idx_1 = np.where(x==1)[0]
    # idx_2 = np.where(x==2)[0]
    # idx_1 = list(idx_1)
    # idx_2 = list(idx_2)
    # idx_1.extend(idx_2)
    # print(idx_1)
    pic()
    # client_pic()
