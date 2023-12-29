from torch import nn
from torch.nn import init
import numpy as np
import scipy.io as sio
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.cluster import KMeans
from mmcv.ops import DeformConv2dPack as DCN

def random_unison(a,b, rstate=None):
    assert len(a) == len(b)
    p = np.random.RandomState(seed=rstate).permutation(len(a))
    return a[p], b[p]

# weights_init
def weights_init(m):
    if isinstance(m, (nn.Conv2d)):
       init.kaiming_normal_(m.weight.data)
    elif isinstance(m, DCN):
       init.kaiming_normal_(m.weight.data)
    #    init.constant_(m.bias.data, 0)


def pca(data, dim = 4):
    nPCs = dim
    X = np.double(np.reshape(data , (data.shape[0]*data.shape[1] , data.shape[2])))
    mu = np.mean(X,axis=0)
    for i in range(data.shape[2]):
        X[:,i] = X[:,i] - mu[i]
    v = np.cov(X.T)
    eig_vals, eig_vecs = np.linalg.eig(v)
    img = np.zeros([data.shape[0], data.shape[1], dim], dtype=np.double)
    for i in range(nPCs):
        img[:,:,i] = np.reshape(np.matmul(X,eig_vecs[:,i]), (data.shape[0], data.shape[1]))
    return img


def loadData(name, num_components=False, standard = False):
    # load data amd mask
    if name == 'ABU_A4':
        dataset = sio.loadmat('./data/ABU_A4.mat')
        data = dataset['data']
        mask = dataset['mask']
    elif name == 'HYDICE':
        dataset = sio.loadmat('./data/HYDICE_Urban.mat') 
        data = dataset['data']
        mask = dataset['mask']
    elif name == 'PC':
        dataset = sio.loadmat('./data/Pavia_Center.mat') 
        data = dataset['data']
        mask = dataset['mask']
    elif name == 'SD':
        dataset = sio.loadmat('./data/San_Diego.mat')
        data = dataset['data']
        mask = dataset['mask']
    else:
        print("NO DATASET")
        exit()
    
    if num_components != False:
        data_PCA = pca(data, num_components)
        data_PCA -= np.amin(data_PCA)
        data_PCA = data_PCA / np.amax(data_PCA)
        return data, data_PCA, mask
    
    return data, mask

def estimate_Nclust(data2d):
# ##estimate the number of clusters###########################
    sse_list = [ ] 
    K = range(1, 30) 
    for k in K: 
        kmeans=KMeans(n_clusters=k,init='k-means++',random_state=20) 
        kmeans.fit(data2d) 
        sse_list.append(kmeans.inertia_)  

    plt.figure(figsize=(4,4)) 
    plt.plot(K, sse_list, 'b.-', linewidth=2)
    # plt.rcParams['figure.figsize'] = [5,5]
    plt.xlabel('The number of clusters',fontsize=12)
    plt.ylabel('SSE',fontsize=12)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    plt.title("The desicion graph of the number of clusters")
    plt.show() 

    Nclust = input("Please input the number of the clusters：") 
    Nclust = int(Nclust)  

    return Nclust

def KClus(data2d, args):

    ##################Clustering#######################
    C_class = KMeans(n_clusters=args.en_dim, random_state=20)
    C_class.fit(data2d)
    Ccenter = C_class.cluster_centers_
    Plabels = C_class.labels_
    Pnum = np.zeros(max(C_class.labels_)+1)
    # Plabels_2D = np.reshape(Plabels, (100,100))
    # plt.imshow(Plabels_2D,cmap='coolwarm')

    for k in range(max(C_class.labels_)+1):
        Pnum[k]=np.sum(Plabels==k)
    # print(np.argmin(Pnum))
    locate_p = np.where(Pnum<0.01*data2d.shape[0])
    Ccenter1 = np.delete(Ccenter, locate_p, axis = 0)
    args.en_dim = len(Ccenter1)
    weight_init = Ccenter1[np.newaxis,np.newaxis]
    
    weight_init1 = np.transpose(weight_init, (2, 3, 0, 1))
    weight_init1 = torch.from_numpy(weight_init1)
    weight_init2 = np.transpose(weight_init, (3, 2, 0, 1))
    weight_init2 = torch.from_numpy(weight_init2)

    return weight_init2, args
 

def accuracy(output, target, mask, draw = False):
    """Computes the precision@k for the specified values of k"""
    # data_2D = data_2D.cuda()
    row, col = mask.shape[0], mask.shape[1]
    res = target-output
    # print(res.shape)
    # anomaly_degree = torch.sqrt(torch.mean(torch.square(res),axis=1))
    anomaly_degree = torch.mean(torch.square(res),axis=1)
    anomaly_map = torch.reshape(anomaly_degree, (row,col))
    anomaly_map = anomaly_map.data.cpu().numpy()
    
    if draw != False:
       plt.figure()
       plt.imshow(anomaly_map,cmap='coolwarm')

    PD, PF, AUC = ROC_AUC(anomaly_map, mask, draw)# coding=utf-8

    return anomaly_map, PD, PF, AUC


def ROC_AUC(anomaly_map, mask, draw = False):
    row, col = anomaly_map.shape
    predict = np.reshape(anomaly_map, (row*col))
    predict = (predict-np.amin(predict))/(np.amax(predict)-np.amin(predict))
    
    mask = np.reshape(mask, (row*col))
   
    fpr, tpr, threshold= roc_curve(mask, predict)
    roc_auc = auc(fpr, tpr)
    # print('roc=',roc_auc)

    if draw != False:
    # draw ROC
       lw = 2
       plt.figure(figsize=(10,10))
       plt.plot(fpr, tpr, color='darkorange',lw=lw, label='ROC curve (area = %0.4f)' % roc_auc) ###假正率为横坐标，真正率为纵坐标做曲线
       plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
       plt.xlim([0.0, 1.0])
       plt.ylim([0.0, 1.05])
       plt.xlabel('False Positive Rate')
       plt.ylabel('True Positive Rate')
       plt.title('ROC curve')
       plt.legend(loc="lower right")
       plt.show()
    
    return tpr, fpr, roc_auc
