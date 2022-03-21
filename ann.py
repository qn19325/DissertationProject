import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision.io import read_image
import pandas as pd
import numpy as np
import os

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = 'cpu'

num_epochs = 100
learning_rate = 0.001
input_size = 8192
hidden_size = 128
output_size = 11

dataset = CustomImageDataset('data.csv', 'trainingData')

class ANN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(ANN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)   # hidden layer
        self.fc2 = nn.Linear(hidden_size, output_size)   # output layer

    def forward(self, x):
        x = F.relu(self.fc1(x))      # activation function
        x = self.fc2(x)             # linear output
        return x

model = ANN(input_size, hidden_size, output_size).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)


for epoch in range(num_epochs):
    for i in range(len(dataset)-1):
        images = dataset[i][0]
        labels = dataset[i][1]
        labels = labels.type(torch.LongTensor)
        # Forward pass
        outputs = model(images)
        print(outputs)
        print(labels)
        loss = criterion(outputs, labels)
        
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print (f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')