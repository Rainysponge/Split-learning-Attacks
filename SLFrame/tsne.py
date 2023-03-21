import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import matplotlib
import os
from collections import defaultdict
import pickle


curpath = ""
epoch = -1
alpha = -1
plt.rcParams.update({
    "font.size": 12,
    "text.usetex": True,
    "font.family": "sans-serif",
})


def plot_embedding(data, label, cn, title, save_path):
    x_min, x_max = np.min(data, 0), np.max(data, 0)
    data = (data - x_min) / (x_max - x_min)
    cs = ["red", "blue", "green", "purple", "gold",
          "cyan", "brown", "black", "gray", "magenta"]
    cs2 = ["magenta", "blue", "green", "purple",
           "gold", "cyan", "brown", "black", "gray", "red"]

    figs, ax = plt.subplots()
    figs.tight_layout(rect=[0, 0.01, 1, 0.95])
    figs.set_figheight(2.4)
    figs.set_figwidth(3.6)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title('Label')
    Q = {}
    for i in range(data.shape[0]):
        s = ax.scatter(data[i, 0], data[i, 1], s=12,
                       marker='.', color=cs2[label[i]])
        Q[label[i]] = s
    Q = dict(sorted(Q.items(), key=lambda x: x[0], reverse=False))
    plt.savefig(
        save_path+"_label.pdf".format(alpha, epoch), format='pdf')

    figs, ax = plt.subplots()
    figs.tight_layout(rect=[0, 0.01, 1, 0.95])
    figs.set_figheight(2.4)
    figs.set_figwidth(3.6)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title('Client')
    Q = {}
    for i in range(data.shape[0]):
        s = plt.scatter(data[i, 0], data[i, 1], s=12,
                        marker='.', color=cs[cn[i]-1])
        Q[cn[i]] = s
    Q = dict(sorted(Q.items(), key=lambda x: x[0], reverse=False))

    plt.savefig(
        save_path+"_client.pdf".format(alpha, epoch), format='pdf')


def process(rootpath):

    for root, dirs, files in os.walk(rootpath):
        tt = root.split('/')[-1]
        if tt == "res" or tt == "pdfs":
            continue
        st = set()
        clients = defaultdict(set)
        for file in files:

            tokens = '.'.join(file.split('.')[0:-1]).split('_')

            print(tokens)
            alpha = tokens[1]
            epoch = tokens[3]
            client = tokens[5]
            st.add((alpha, epoch))
            clients[(alpha, epoch)].add(client)
            print(alpha, epoch, client)

        for aa, ee in st:

            dic = defaultdict(list)
            for clinet in clients[(aa, ee)]:
                print(aa, ee, clinet)
                fname = root+"/A_{}_E_{}_C_{}.txt".format(aa, ee, clinet)
                print("fname", fname)
                f = open(fname, "r")
                lines = f.readlines()

                for line in lines:
                    ll = line.strip().split(maxsplit=2)
                    c, l, a = ll
                    c = int(c)
                    l = int(l)
                    a = list(map(float, a.strip('[').strip(']').split(',')))
                    # print(a)
                    dic[(aa, ee, "data")].append(a)
                    dic[(aa, ee, "label")].append(l)
                    dic[(aa, ee, "cn")].append(c)

            print("Finishing data loading(A_{}_E_{})".format(aa, ee))
            paradigm = root.split('/')[-1]
            curpath = "./model_save/acts/res/"+paradigm
            print(curpath)
            print("start", aa, ee)
            alpha, epoch = aa, ee
            data = np.array(dic[(aa, ee, "data")])
            print(data.shape)
            tsne = TSNE(n_components=2, init='pca', random_state=0)
            result = tsne.fit_transform(data)
            print('result.shape', result.shape)
            save_p = curpath+"_A_{}_E_{}_2d.pkl".format(aa, ee)
            with open(save_p, "wb+") as f:
                tp = (paradigm, aa, ee, result,
                      dic[(aa, ee, "label")], dic[(aa, ee, "cn")])
                pickle.dump(tp, f)
            print("done", aa, ee)


def draw(from_path, to_path):
    for root, dirs, files in os.walk(from_path):
        for file in files:
            qaq = file.rsplit('.', 1)[0]
            with open(root+'/'+file, "rb") as f:
                paradigm, aa, ee, result, label, cn = pickle.load(f)
            if paradigm != "PSL" or int(ee) % 100 != 0:
                continue
            plot_embedding(result, label, cn, None, to_path+'/'+qaq)


# rootpath = r"./model_save/acts"
# process(rootpath)
fp, tp = "./model_save/acts/res", "./model_save/acts/pdfs"
draw(fp, tp)

# for e in range(5):

#     epoch=e*5
#     data=[]
#     label=[]
#     c=[]
#     QAQ=0
#     for c in range(5):
#         f=open(rootpath+"/C{}E{}.txt".format(c+1,epoch),"r")
#         lines=f.readlines()
#         for line in lines:
#             ll=line.strip().split(maxsplit=2)
#             c,l,a = ll
#             c=int(c)
#             l=int(l)
#             a=list(map(float,a.strip('[').strip(']').split(',')))
#            # print(a)
#             data.append(a)
#             label.append(l)
#             cc.append(c)
#             QAQ+=1
#             print(QAQ)

#     data=np.array(data)
#     tsne=TSNE(n_components=2, init='pca', random_state=0)
#     result = tsne.fit_transform(data)
#     print('result.shape', result.shape)
#     plot_embedding(result, label,cc,'')

# # else :
# #     c,l=1,3
# #     arr=np.ones(28*28)
# #     a=str(list(arr))
# #     print(a)
# #     f=open("model_save/a.txt","w")
# #     f.write("{} {} {}\n".format(c,l,a))
# #     arr=arr/3
# #     a=str(list(arr))
# #     c,l=2,4
# #     f.write("{} {} {}\n".format(c,l,a))
