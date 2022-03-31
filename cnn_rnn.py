import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision
import pandas as pd
import numpy as np

class ImageDataLoader(Dataset):
    def __init__(self, dir_=None):
        self.data_df = pd.read_csv('data.csv')
        self.dataset_len = len(self.data_df) # read the number of len of your csv files
    def __getitem__(self, idx):
        # load the next image
        f_name_t = self.data_df['Filename'][idx]
        f_name_tp1 = self.data_df['Filename'][idx+1]
        label = self.data_df['Label'][idx]
        label = label.astype(np.float32) 
        label = np.true_divide(label, 10)
        img_t = torchvision.io.read_image('training/{}'.format(f_name_t))
        img_tp1 = torchvision.io.read_image('training/{}'.format(f_name_tp1))
        img_t = img_t.float().div_(255.0)
        img_tp1 = img_tp1.float().div_(255.0)
        return img_t, img_tp1, label
    def __len__(self):
        return self.dataset_len - 1

dataset = ImageDataLoader()
dataloader = DataLoader(dataset, shuffle=True, batch_size=8)

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 8, 1, 1),
            nn.ReLU(),
            nn.Conv2d(8, 16, 1, 1),
            nn.ReLU(),
            nn.Flatten()
        )
        self.fc1 = nn.Linear(65536, 128)
    def forward(self, x):
        state = self.cnn(x)
        state = self.fc1(state)
        # print('size of the state after CNN ',state.size())
        return state

encoder = Encoder()

batch_size = 8
num_epochs = 1000
learning_rate = 0.001
input_size = 128
output_size = 1
sequence_length = 2
hidden_size = 128
num_layers = 1

class RNN(nn.Module):
    def __init__(self, batch_size, input_size, hidden_size, num_layers, output_size):
        super(RNN, self).__init__()
        self.batch_size = batch_size
        self.input_size = input_size
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.rnn = nn.RNN(self.input_size, self.hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(self.hidden_size, self.output_size)
    def init_hidden(self):
        return (torch.zeros(self.num_layers, self.batch_size, self.hidden_size))
    def forward(self, x):
        self.batch_size = x.size(0)
        self.hidden = self.init_hidden()
        out, self.hidden = self.rnn(x, self.hidden)
        out = self.fc(out)
        return out

model = RNN(batch_size=batch_size, input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, output_size=output_size)

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

n_total_steps = len(dataloader)
for epoch in range(num_epochs):
    runningLoss = 0
    for i, (image1, image2, label) in enumerate(dataloader):
        output1 = encoder(image1)
        output2 = encoder(image2)
        # print('o1 shape:', output1.size())
        batch_size1 = len(output1)
        batch_size2 = len(output2)
        output1 = output1.reshape(batch_size1,1,-1)
        output2 = output2.reshape(batch_size2,1,-1)
        # print('o1 after shape:', output1.size())
        seq = torch.cat((output1, output2.detach()), dim=1)
        # print('seq size: ', seq.size())
        label = (label.float())
        # Forward pass
        outputs = model(seq)
        loss = criterion(outputs[:,-1].squeeze(), label)
        # print(outputs[:,-1].squeeze())
        # print(label)
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        runningLoss += loss
    if epoch%10 == 0:
        print (f'Epoch [{epoch+1}/{num_epochs}], Loss: {runningLoss:.4f}')
    runningLoss = 0