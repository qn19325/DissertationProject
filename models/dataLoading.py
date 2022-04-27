import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
  
class ImageDataLoader(Dataset):
  def __init__(self, dir_=None):
    self.data_df = pd.read_csv('trainingData/basicEnv/256x256.csv')
    self.dataset_len = len(self.data_df) # read the number of len of your csv files
  def __getitem__(self, idx):
    # load the next image
    f_name_t = self.data_df['Filename'][idx]
    f_name_tp1 = self.data_df['Filename'][idx+1]
    label = self.data_df['Label'][idx]
    label = label.astype(np.float32) 
    label = np.true_divide(label, 20)
    img_t = torchvision.io.read_image('trainingData/basicEnv/256x256/{}'.format(f_name_t))
    img_tp1 = torchvision.io.read_image('trainingData/basicEnv/256x256/{}'.format(f_name_tp1))
    img_t = img_t.float().div_(255.0)
    img_tp1 = img_tp1.float().div_(255.0)
    # # invert pixel values
    # img_t = 1 - img_t
    # img_tp1 = 1 - img_tp1
    # # crop image
    # img_t = torchvision.transforms.functional.crop(img_t, 0, 0, 32, 64)
    # img_tp1 = torchvision.transforms.functional.crop(img_tp1, 0, 0, 32, 64)
    return img_t, img_tp1, label
  def __len__(self):
    return self.dataset_len - 1 

dataset = ImageDataLoader()
data_loader = DataLoader(dataset, 1, shuffle=False)

def imshow(img):
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

iterableData = iter(data_loader)
images, _, _ = next(iterableData)
images = images.reshape(-1,1,256,256)
torchvision.utils.save_image(images, "plots/original_datapoint.png")
images = 1 - images
torchvision.utils.save_image(images, "plots/inverted_datapoint.png")
images = torchvision.transforms.functional.crop(images, 0, 0, 128, 256)
torchvision.utils.save_image(images, "plots/final_datapoint.png")

