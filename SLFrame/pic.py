import numpy as np
import pandas as pd
import torch
import seaborn
from matplotlib import pyplot as plt
import seaborn as sns
import re
from queue import Queue

if __name__ == "__main__":
    q = Queue(maxsize=0)
    q.put(1)
    q.put(2)
    while not q.empty():
        print(q.get())
    # with open("D:/Split-learning-Attacks/SABuf2/Split-learning-Attacks/SLFrame/log.txt", "r") as f:
    #     content = f.read()
    #
    # p = re.compile("phase=validation acc=(.*?) loss")
    # matchObj = re.findall(p, content)
    #
    # acc = [(float(matchObj[2*i]) + float(matchObj[2*i+1])) / 2 for i in range(int(len(matchObj)) // 2)]
    # x = [i for i in range(len(acc))]
    # data = np.array([x, acc])
    #
    # df = pd.DataFrame(data.T, columns=["turn", "acc"])
    # df["turn"] = df["turn"].astype(int)
    # df["acc"] = df["acc"].astype(float)
    # print(data)
    # print(df.dtypes)
    # sns.lineplot(x="turn", y="acc", data=df)
    # plt.show()
