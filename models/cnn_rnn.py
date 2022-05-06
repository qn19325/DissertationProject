import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision
import pandas as pd
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

config = dict(
    epochs=1000,
    train_batch_size=8,
    test_batch_size=1,
    learning_rate=0.001,
    input_size = 128,
    hidden_size = 128,
    output_size = 1,
    sequence_length = 2,
    num_layers = 1,
    dataset="basic256x256",
    architecture="CNN/RNN")

def model_pipeline(hyperparameters):

    # make the model, data, and optimization problem
    encoder, model, train_loader, test_loader, criterion, optimizer = make(config)
    print(model)

    # and use them to train the model
    main(encoder, model, train_loader, test_loader, criterion, optimizer, config)

    return model

def make(config):
    # Make the data
    dataset = get_data()
    train_loader = make_loader(dataset, batch_size=config.get("train_batch_size"))
    test_loader = make_loader(dataset, batch_size=config.get("test_batch_size"))

    # Make the CNN encoder
    encoder = Encoder().to(device)
    # Make the RNN model
    model = RNN(config).to(device)

    # Make the loss and optimizer
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), 
                                 lr=config.get("learning_rate"))
    
    return encoder, model, train_loader, test_loader, criterion, optimizer

def get_data():
    full_dataset = ImageDataLoader()
    
    return full_dataset

def make_loader(dataset, batch_size):
    loader = DataLoader(dataset=dataset, 
                        batch_size=batch_size, 
                        shuffle=True)
    return loader

class ImageDataLoader(Dataset):
    def __init__(self, dir_=None):
        self.data_df = pd.read_csv('trainingData-basicEnv/256x256.csv')
        self.dataset_len = len(self.data_df) # read the number of len of your csv files
    def __getitem__(self, idx):
        # load the next image
        f_name_t = self.data_df['Filename'][idx]
        f_name_tp1 = self.data_df['Filename'][idx+1]
        label = self.data_df['Label'][idx]
        label = label.astype(np.float32) 
        label = np.true_divide(label, 20)
        img_t = torchvision.io.read_image('trainingData-basicEnv/256x256/{}'.format(f_name_t))
        img_tp1 = torchvision.io.read_image('trainingData-basicEnv/256x256/{}'.format(f_name_tp1))
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
        self.fc1 = nn.Linear(1048576, 128)
    def forward(self, x):
        state = self.cnn(x)
        state = self.fc1(state)
        return state   
# class Encoder(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.conv1 = nn.Conv2d(1, 16, 8, 2)
#         self.relu1 = nn.ReLU()
#         self.conv2 = nn.Conv2d(16, 32, 8, 2)
#         self.relu2 = nn.ReLU()
#         self.fc = nn.Linear(32, 16)
#     def forward(self, x_t):
#         conv1_out = self.conv1(x_t)
#         # print('conv1_out size: ', conv1_out.size())
#         relu1_out = self.relu1(conv1_out)
#         conv2_out = self.conv2(relu1_out)
#         relu2_out = self.relu2(conv2_out)
#         # print('l23_act_out size: ', l23_act_out.size())
#         feature_vector = relu2_out.mean(dim=(-2, -1)) # global average pooling
#         # print('feature_vector L23 size: ', feature_vector.size())
#         fc_out = self.fc(feature_vector)
#         # print('fc_out size: ', fc_out.size())
#         return fc_out

class RNN(nn.Module):
    def __init__(self, config):
        super(RNN, self).__init__()
        self.input_size = config.get("input_size")
        self.num_layers = config.get("num_layers")
        self.hidden_size = config.get("hidden_size")
        self.output_size = config.get("output_size")
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
    
def main(encoder, model, train_loader, test_loader, criterion, optimizer, config):

    # Run training and track with wandb
    example_ct = 0
    batch_ct = 0
    for epoch in tqdm(range(config.get("epochs"))):
      for _, (images1, images2, labels) in enumerate(train_loader):
        loss = train(images1, images2, labels, encoder, model, optimizer, criterion, epoch)
        example_ct += len(images1)
        batch_ct += 1

        # Report metrics every 25th batch
        if ((batch_ct + 1) % 25) == 0:
          print(f"Epoch: " + str(epoch) + f", Batch: " + str(batch_ct).zfill(4) + f", Example: " + str(example_ct).zfill(5) + f", Loss: {loss:.3f}")

    #   if epoch % 5 == 0:
    #       test(test_loader, encoder, model, epoch)

    #   if epoch % 25 == 0:
    #     EPOCH = epoch
    #     PATH = "gdrive/MyDrive/models/cnn_rnn_" + str(epoch) + ".pt"
    #     LOSS = loss
    #     torch.save({
    #                 'epoch': EPOCH,
    #                 'model_state_dict': model.state_dict(),
    #                 'encoder_state_dict': encoder.state_dict(),
    #                 'optimizer_state_dict': optimizer.state_dict(),
    #                 'loss': LOSS,
    #                 }, PATH)
                
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

def train(images1, images2, labels, encoder, model, optimizer, criterion, epoch):
  model.train()

  images1, images2, labels = images1.to(device), images2.to(device), (labels.float()).to(device)
  toShow1 = images1.reshape(-1,1,256,256)
  imshow(images1)
#   print('images1:', images1.size())
#   print(images1)
#   print('images2:', images2.size())
  # Forward pass ➡
  # pass to encoder
  output1 = encoder(images1)
  output2 = encoder(images2)
#   print('output1:', output1.size())
#   print('output2:', output2.size())
  # pass to RNN
  batch_size1 = len(output1)
  batch_size2 = len(output2)

  output1 = output1.reshape(batch_size1,1,-1)
  output2 = output2.reshape(batch_size2,1,-1)
#   print('output1:', output1.size())
#   print('output2:', output2.size())
  
  seq = torch.cat((output1, output2.detach()), dim=1)
#   print("seq:", seq.size())
#   print(seq)

  outputs = model(seq.to(device))
  if epoch % 10 == 0:
    print('labels:', labels)
    print('outputs:', outputs[:,-1].squeeze())
  loss = criterion(outputs[:,-1].squeeze(), labels.float())

  # Backward pass ⬅
  optimizer.zero_grad()
  loss.backward()

  # Step with optimizer
  optimizer.step()

  return loss


model = model_pipeline(config)