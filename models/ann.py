import torch
import torch.nn as nn
import torchvision
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

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
        # For ANN
        img_t = img_t.flatten().float().div_(255.0)
        img_tp1 = img_tp1.flatten().float().div_(255.0)
        input_seq = torch.cat((img_t, img_tp1), dim=0)
        return input_seq, label
    def __len__(self):
        return self.dataset_len - 1

dataset = ImageDataLoader()
dataloader = DataLoader(dataset, shuffle=False, batch_size=8)

batch_size = 4
input_size = 8192
hidden_size = 128
output_size = 1
num_epochs = 100
learning_rate = 0.05

class ANN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(ANN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)   # hidden layer
        self.fc2 = nn.Linear(hidden_size, output_size)   # output layer

    def forward(self, x):
        x = F.relu(self.fc1(x))      # activation function
        x = self.fc2(x)             # linear output
        return x

model = ANN(input_size, hidden_size, output_size)

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

for epoch in range(num_epochs):
    correct = 0 
    for i, (images,labels) in enumerate(dataloader):
        labels = (labels.float())
        # Forward pass
        outputs = model(images)
        loss = criterion(outputs[:,-1], labels)
        
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        correct += (outputs == labels).float().sum()
    accuracy = 100 * correct / len(dataloader)
    print (f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}, Accuracy = {accuracy}')