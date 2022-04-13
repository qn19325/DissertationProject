import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision
import pandas as pd
import numpy as np
import wandb
from tqdm.notebook import tqdm

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

config = dict(
    epochs=1000,
    batch_size=8,
    learning_rate=0.001,
    input_size = 128,
    hidden_size = 128,
    output_size = 1,
    sequence_length = 2,
    num_layers = 1,
    dataset="basic64x64",
    architecture="CNN/RNN")

def model_pipeline(hyperparameters):

    # tell wandb to get started
    with wandb.init(project="individual_project", config=hyperparameters):
      # access all HPs through wandb.config, so logging matches execution!
      config = wandb.config

      # make the model, data, and optimization problem
      encoder, model, train_loader, criterion, optimizer = make(config)
      print(model)

      # and use them to train the model
      train(encoder, model, train_loader, criterion, optimizer, config)

    return model

def make(config):
    # Make the data
    train = get_data(train=True)
    train_loader = make_loader(train, batch_size=config.batch_size)

    # Make the CNN encoder
    encoder = Encoder().to(device)
    # Make the RNN model
    model = RNN(config).to(device)

    # Make the loss and optimizer
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), 
                                 lr=config.learning_rate)
    
    return encoder, model, train_loader, criterion, optimizer

def get_data(slice=5, train=True):
    full_dataset = ImageDataLoader()
    
    return full_dataset

def make_loader(dataset, batch_size):
    loader = DataLoader(dataset=dataset, 
                        batch_size=batch_size, 
                        shuffle=True)
    return loader

class ImageDataLoader(Dataset):
    def __init__(self, dir_=None):
        self.data_df = pd.read_csv('gdrive/MyDrive/64x64.csv')
        self.dataset_len = len(self.data_df) # read the number of len of your csv files
    def __getitem__(self, idx):
        # load the next image
        f_name_t = self.data_df['Filename'][idx]
        f_name_tp1 = self.data_df['Filename'][idx+1]
        label = self.data_df['Label'][idx]
        label = label.astype(np.float32) 
        label = np.true_divide(label, 20)
        img_t = torchvision.io.read_image('gdrive/MyDrive/64x64/{}'.format(f_name_t))
        img_tp1 = torchvision.io.read_image('gdrive/MyDrive/64x64/{}'.format(f_name_tp1))
        img_t = img_t.float().div_(255.0)
        img_tp1 = img_tp1.float().div_(255.0)
        return img_t, img_tp1, label
    def __len__(self):
        return self.dataset_len - 1

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
        return state

class RNN(nn.Module):
    def __init__(self, config):
        super(RNN, self).__init__()
        self.batch_size = config.batch_size
        self.input_size = config.input_size
        self.num_layers = config.num_layers
        self.hidden_size = config.hidden_size
        self.output_size = config.output_size
        self.rnn = nn.RNN(self.input_size, self.hidden_size, self.num_layers, batch_first=True)
        self.fc = nn.Linear(self.hidden_size, self.output_size)
    def init_hidden(self):
        return (torch.zeros(self.num_layers, self.batch_size, self.hidden_size).to(device))
    def forward(self, x):
        self.batch_size = x.size(0)
        self.hidden = self.init_hidden()
        out, self.hidden = self.rnn(x, self.hidden)
        out = self.fc(out)
        return out

def train(encoder, model, loader, criterion, optimizer, config):
    # Tell wandb to watch what the model gets up to: gradients, weights, and more!
    wandb.watch(model, criterion, log="all", log_freq=10)

    # Run training and track with wandb
    total_batches = len(loader) * config.epochs
    example_ct = 0  # number of examples seen
    batch_ct = 0
    for epoch in tqdm(range(config.epochs)):
        for _, (images1, images2, labels) in enumerate(loader):

            loss = train_batch(images1, images2, labels, encoder, model, optimizer, criterion)
            example_ct += len(images1)
            batch_ct += 1

            # Report metrics every 25th batch
            if ((batch_ct + 1) % 25) == 0:
                train_log(loss, example_ct, epoch)


def train_batch(images1, images2, labels, encoder, model, optimizer, criterion):
    images1, images2, labels = images1.to(device), images2.to(device), (labels.float()).to(device)
    
    # Forward pass ➡
    # pass to encoder
    output1 = encoder(images1)
    output2 = encoder(images2)
    # pass to RNN
    batch_size1 = len(output1)
    batch_size2 = len(output2)

    output1 = output1.reshape(batch_size1,1,-1)
    output2 = output2.reshape(batch_size2,1,-1)
    
    seq = torch.cat((output1, output2.detach()), dim=1)

    outputs = model(seq.to(device))
    loss = criterion(outputs[:,-1].squeeze(), labels.float())

    # Backward pass ⬅
    optimizer.zero_grad()
    loss.backward()

    # Step with optimizer
    optimizer.step()

    return loss

def train_log(loss, example_ct, epoch):
    # Where the magic happens
    wandb.log({"epoch": epoch, "loss": loss}, step=example_ct)
    print(f"Loss after " + str(example_ct).zfill(5) + f" examples: {loss:.3f}")

model = model_pipeline(config)