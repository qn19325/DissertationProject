import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision
import torchvision.transforms as transforms
import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np
import random

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = 'cpu'

class ImageDataLoader(Dataset):
    def __init__(self, dir_=None):
        self.data_df = pd.read_csv('data.csv')
        self.dataset_len = len(self.data_df) # read the number of len of your csv files
    def __getitem__(self, idx):
        # load the next image
        f_name_t = self.data_df['Filename'][idx]
        f_name_tp1 = self.data_df['Filename'][idx+1]
        label = self.data_df['Label'][idx]
        img_t = torchvision.io.read_image('trainingData/{}'.format(f_name_t))
        img_tp1 = torchvision.io.read_image('trainingData/{}'.format(f_name_tp1))
        img_t = img_t.flatten(1,-1).float().div_(255.0)
        img_tp1 = img_tp1.flatten(1,-1).float().div_(255.0)
        input_seq = torch.cat((img_t, img_tp1), dim=0)
        return input_seq, label
    def __len__(self):
        return self.dataset_len - 1

dataset = ImageDataLoader()
dataloader = DataLoader(dataset, shuffle=False, batch_size=8)

def imshow(img):
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))

imagesList = []
for idx, (images, labels) in enumerate(dataloader):
  images = images.reshape(-1,1,64,64)
  imagesList.append(images)

randomBatch = random.randrange(0,len(imagesList))
print('Batch:', randomBatch)
imshow(torchvision.utils.make_grid(imagesList[randomBatch]))

num_epochs = 100
learning_rate = 0.001
input_size = 4096
output_size = 1
sequence_length = 2
hidden_size = 128
num_layers = 2

class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(RNN, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size) 
        out, _ = self.rnn(x, h0)  
        # out = out[:, -1]
        out = self.fc(out)
        return out

model = RNN(input_size, hidden_size, num_layers, output_size)

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

def get_accuracy(logit, target, batch_size):
    ''' Obtain accuracy for training round '''
    corrects = (torch.max(logit, 1)[1].view(target.size()).data == target.data).sum()
    accuracy = 100.0 * corrects/batch_size
    return accuracy.item()

n_total_steps = len(dataloader)
for epoch in range(num_epochs):
    correct = 0
    for i, (images, labels) in enumerate(dataloader):
        labels = (labels.float()).to(device)
        # Forward pass
        outputs = model(images)
        loss = criterion(outputs[:,-1].flatten(), labels)
        print(labels)
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        correct += (outputs == labels).float().sum()
    accuracy = 100 * correct / len(dataloader)
    print (f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}, Accuracy = {accuracy}')
