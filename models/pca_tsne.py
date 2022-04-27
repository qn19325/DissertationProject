import torch
from torch.utils.data import Dataset, DataLoader
import torchvision
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import time
import seaborn as sns

class ImageDataLoader(Dataset):
    def __init__(self, dir_=None):
        self.data_df = pd.read_csv('trainingData/verticalEnv/64x64.csv')
        self.dataset_len = len(self.data_df) # read the number of len of your csv files
    def __getitem__(self, idx):
        # load the next image
        f_name_t = self.data_df['Filename'][idx]
        f_name_tp1 = self.data_df['Filename'][idx+1]
        label = self.data_df['Label'][idx]
        label = label.astype(np.float32) 
        label = np.true_divide(label, 20)
        label1 = self.data_df['Label'][idx+1]
        label1 = label1.astype(np.float32) 
        label1 = np.true_divide(label1, 20)
        img_t = torchvision.io.read_image('trainingData/verticalEnv/64x64/{}'.format(f_name_t))
        img_tp1 = torchvision.io.read_image('trainingData/verticalEnv/64x64/{}'.format(f_name_tp1))
        img_t = img_t.float().div_(255.0)
        img_tp1 = img_tp1.float().div_(255.0)
        img_t = 1 - img_t
        img_t = torchvision.transforms.functional.crop(img_t, 0, 0, 32, 64)
        img_tp1 = 1 - img_tp1
        img_tp1 = torchvision.transforms.functional.crop(img_tp1, 0, 0, 32, 64)
        return img_t, img_tp1, label, label1
    def __len__(self):
        return self.dataset_len - 1

dataset = ImageDataLoader()
loader = DataLoader(dataset, batch_size=1002, shuffle=False)

def images(data_loader): ### DOESNT APPEAR TO WORK ###
    x_t, x_tp1, a_t, a_tp1 = next(iter(data_loader))
    X = x_t.flatten(1,-1).squeeze().numpy() #-2,-1
    print(X.shape)
    x_t, x_tp1, a_t, a_tp1 = x_t.numpy(), x_tp1.numpy(), a_t.numpy(), a_tp1.numpy()
    return X, a_t

def concatenate(data_loader): ### DOESNT APPEAR TO WORK ###
    x_t, x_tp1, a_t, a_tp1 = next(iter(data_loader))
    X = torch.cat((x_t.flatten(-2,-1).squeeze(), x_tp1.flatten(-2,-1).squeeze()), dim=1)
    print(X.shape)
    x_t, x_tp1, a_t, a_tp1 = x_t.numpy(), x_tp1.numpy(), a_t.numpy(), a_tp1.numpy()
    return X, a_t

def difference(data_loader):
    x_t, x_tp1, a_t, a_tp1 = next(iter(data_loader))
    X = (x_t-x_tp1).squeeze().flatten(-2,-1)
    print(X.shape)
    x_t, x_tp1, a_t, a_tp1 = x_t.numpy(), x_tp1.numpy(), a_t.numpy(), a_tp1.numpy()
    return X, a_t

def pca(dir, env, X, a_t):
    pca = PCA(n_components=3)
    Xt = pca.fit_transform(X)
    print('input EVR', pca.explained_variance_ratio_)
    plot = plt.scatter(Xt[:,0], Xt[:,1], c=a_t)
    plt.xlabel('component 1')
    plt.ylabel('component 2')
    plt.legend(handles=plot.legend_elements()[0], labels=list(a_t))
    plt.suptitle("2 Component PCA - " + env)
    plt.show()
    # plt.savefig("plots/pca_tsne/" + dir + env + "-PCA-alteredData", dpi=300)

def tsne(dir, env, X, a_t):
    time_start = time.time()
    tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
    tsne_results = tsne.fit_transform(X)
    print('t-SNE done! Time elapsed: {} seconds'.format(time.time()-time_start))
    df = pd.DataFrame()
    df['tsne-2d-one'] = tsne_results[:,0]
    df['tsne-2d-two'] = tsne_results[:,1]
    plt.figure(figsize=(16,10))
    sns.scatterplot(
        x="tsne-2d-one", y="tsne-2d-two",
        hue=a_t,
        palette=sns.color_palette("hls", 11),
        data=df,
        legend="full"
    )
    plt.suptitle("t-SNE 2D - " + env)
    # plt.show()
    plt.savefig("plots/pca_tsne/" + dir + env + "-TSNE-alteredData", dpi=300)

# X, a_t = images(loader)
# X, a_t = concatenate(loader)
# X, a_t = difference(loader)
# tsne("difference/", "verticalEnv", X, a_t)
# pca("", X, a_t)

