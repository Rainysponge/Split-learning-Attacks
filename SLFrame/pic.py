import numpy as np
import pandas as pd
import torch
import seaborn
from matplotlib import pyplot as plt
import seaborn as sns
import re
from queue import Queue

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
    with open("D:/Split-learning-Attacks/SABuf2/Split-learning-Attacks/SLFrame/log.txt", "r") as f:
        content = f.read()

    p = re.compile("phase=validation acc=(.*?) loss")
    matchObj = re.findall(p, content)

    acc = [(float(matchObj[i*3]) + float(matchObj[3*i+1]) + float(matchObj[3*i+3])) / 3  for i in range(int(len(matchObj)) // 3)]
    x = [i for i in range(len(acc))]
    data = np.array([x, acc])

    df = pd.DataFrame(data.T, columns=["turn", "acc"])
    df["turn"] = df["turn"].astype(int)
    df["acc"] = df["acc"].astype(float)
    print(data)
    print(df.dtypes)
    sns.lineplot(x="turn", y="acc", data=df)
    plt.show()
