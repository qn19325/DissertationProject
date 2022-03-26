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

class ImageDataLoader(Dataset):
    def __init__(self, dir_=None):
        self.data_df = pd.read_csv('gdrive/MyDrive/data.csv')
        self.dataset_len = len(self.data_df) # read the number of len of your csv files
    def __getitem__(self, idx):
        # load the next image
        f_name_t = self.data_df['Filename'][idx]
        f_name_tp1 = self.data_df['Filename'][idx+1]
        label = self.data_df['Label'][idx]
        img_t = torchvision.io.read_image('gdrive/MyDrive/trainingData/{}'.format(f_name_t))
        img_tp1 = torchvision.io.read_image('gdrive/MyDrive/trainingData/{}'.format(f_name_tp1))
        img_t = img_t.float().div_(255.0)
        img_tp1 = img_tp1.float().div_(255.0)
        return img_t, img_tp1, label
    def __len__(self):
        return self.dataset_len - 1

dataset = ImageDataLoader()
dataloader = DataLoader(dataset, shuffle=False, batch_size=8)

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 16, 1, 1),
            nn.ReLU(),
            nn.Conv2d(16, 32, 1, 1),
            nn.ReLU(),
            nn.Flatten()
        )
    def forward(self, x):
        state = self.cnn(x)
        # print('size of the state after CNN ',state.size())
        return state

encoder = Encoder()

num_epochs = 10
learning_rate = 0.001
input_size = 131072
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

n_total_steps = len(dataloader)
for epoch in range(num_epochs):
    for i, (image1, image2, label) in enumerate(dataloader):
        image1, image2, label = next(iter(dataloader))
        output = encoder(image1)
        output1 = encoder(image2)
        output = output.reshape(8,1,-1)
        output1 = output1.reshape(8,1,-1)
        seq = torch.cat((output, output1), dim=1)
        label = (label.float())

        # Forward pass
        outputs = model(seq)
        loss = criterion(outputs[:,-1], label)
        
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if (i+1) % 10 == 0:
            print (f'Epoch [{epoch+1}/{num_epochs}] | Step [{i+1}/{n_total_steps}] | Loss: {loss.item():.4f}')