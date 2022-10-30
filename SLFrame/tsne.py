import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import matplotlib
import os
from collections import defaultdict

rootpath=r"./model_save/acts/PSL"
curpath=""
epoch=-1
alpha=-1
plt.rcParams.update({
    "font.size": 12,
    "text.usetex": True,
    "font.family": "sans-serif",
    })


def plot_embedding(data, label, cn, title):
    x_min, x_max = np.min(data, 0), np.max(data, 0)
    data = (data - x_min) / (x_max - x_min)
    cs = [plt.cm.Set1(i) for i in range(10)]
    cs.append(plt.cm.Set2(0))
    
    
    
    
    figs, ax = plt.subplots()
    figs.tight_layout(rect=[0.05, 0.06, 1, 0.98])
    figs.set_figheight(2.4)
    figs.set_figwidth(3.6)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title('Label')
    Q={}
    for i in range(data.shape[0]):
        s=ax.scatter(data[i, 0], data[i, 1],s=12,marker='.',color=cs[label[i]])
        Q[label[i]]=s
    Q=dict(sorted(Q.items(), key=lambda x: x[0], reverse=False))
    plt.savefig(curpath+"_A_{}_E_{}_label.pdf".format(alpha,epoch),format='pdf')
    
    
    figs, ax = plt.subplots()
    figs.tight_layout(rect=[0.05, 0.06, 1, 0.98])
    figs.set_figheight(2.4)
    figs.set_figwidth(3.6)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title('Client')
    Q={}
    for i in range(data.shape[0]):
        s=plt.scatter(data[i, 0], data[i, 1],s=12,marker='.',color=cs[cn[i]])
        Q[cn[i]]=s
    Q = dict(sorted(Q.items(), key=lambda x: x[0], reverse=False))
    
    plt.savefig(curpath+"_A_{}_E_{}_client.pdf".format(alpha,epoch),format='pdf')



for root,dirs, files in os.walk(rootpath):
    dic=defaultdict(list)
    st= set()
    for file in files:
        
        tokens='.'.join(file.split('.')[0:2]).split('_')
        print(tokens)
        alpha=tokens[1]
        epoch=tokens[4]
        client=tokens[2]
        st.add((alpha,epoch))
        print(alpha,epoch,client)
        f=open(root+'/'+file,"r")
        
        lines=f.readlines()
        for line in lines:
            ll=line.strip().split(maxsplit=2)
            c,l,a = ll
            c=int(c)
            l=int(l)
            a=list(map(float,a.strip('[').strip(']').split(',')))
           # print(a)
            dic[(alpha,epoch,"data")].append(a)
            dic[(alpha,epoch,"label")].append(l)
            dic[(alpha,epoch,"cn")].append(c)
            
    print("Finishing data loading.")
    curpath=root
    for a,e in st:
        alpha,epoch=a,e
        data=np.array(dic[(a,e,"data")])
        print(data.shape)
        tsne=TSNE(n_components=2, init='pca', random_state=0)
        result = tsne.fit_transform(data)
        print('result.shape', result.shape)
        plot_embedding(result, dic[(a,e,"label")],dic[(a,e,"cn")],'')
        print("done",a,e)
   
        



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

