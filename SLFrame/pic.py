import numpy as np
import pandas as pd
import torch
import seaborn
from matplotlib import pyplot as plt
import seaborn as sns
import re

if __name__ == "__main__":
    with open("./german 3 15 (2).txt", "r") as f:
        content = f.read()

    p = re.compile("phase=validation acc=(.*?) loss")
    matchObj = re.findall(p, content)

    acc = [matchObj[i * 2] for i in range(int(len(matchObj) // 2))]
    x = [i for i in range(len(acc))]
    data = np.array([x, acc])
    df = pd.DataFrame(data.T, columns=["turn", "acc"])
    df["turn"] = df["turn"].astype(int)
    df["acc"] = df["acc"].astype(float)
    # print(data)
    print(df.dtypes)
    sns.lineplot(x="turn", y="acc", data=df)
    plt.show()
