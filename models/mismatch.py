import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from mpl_toolkits.axes_grid1 import make_axes_locatable

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

config = dict(
    epochs=1000,
    train_batch_size=8,
    test_batch_size=1,
    mismatch_batch_size=1,
    learning_rate=0.001,
    input_size = 32,
    hidden_size = 128,
    output_size = 1,
    sequence_length = 2,
    num_layers = 1,
    dataset="basic64x64",
    architecture="CNN/RNN")

class ImageDataLoader(Dataset):
  def __init__(self, dir_=None):
    self.data_df = pd.read_csv('trainingData/basicEnv/64x64.csv')
    self.dataset_len = len(self.data_df) # read the number of len of your csv files
  def __getitem__(self, idx):
    # load the next image
    f_name_t = self.data_df['Filename'][idx]
    f_name_tp1 = self.data_df['Filename'][idx+1]
    label = self.data_df['Label'][idx]
    label = label.astype(np.float32) 
    label = np.true_divide(label, 20)
    img_t = torchvision.io.read_image('trainingData/basicEnv/64x64/{}'.format(f_name_t))
    img_tp1 = torchvision.io.read_image('trainingData/basicEnv/64x64/{}'.format(f_name_tp1))
    img_t = img_t.float().div_(255.0)
    img_tp1 = img_tp1.float().div_(255.0)
    # invert pixel values
    img_t = 1 - img_t
    img_tp1 = 1 - img_tp1
    # crop image
    img_t = torchvision.transforms.functional.crop(img_t, 0, 0, 32, 64)
    img_tp1 = torchvision.transforms.functional.crop(img_tp1, 0, 0, 32, 64)
    return img_t, img_tp1, label
  def __len__(self):
    return self.dataset_len - 1 

class Encoder(nn.Module):
  def __init__(self):
    super(Encoder, self).__init__()
    self.conv1 = nn.Conv2d(1, 8, 4, 1)
    self.relu1 = nn.ReLU()
    self.conv2 = nn.Conv2d(8, 16, 4, 1)
    self.relu2 = nn.ReLU()
    self.flatten = nn.Flatten() # can use global average pooling
    self.fc = nn.Linear(24128, 32)
            
    self.hook = {'conv1_out': [],'relu1_out': [],'conv2_out': [],'relu2_out': [],'fc_out': []}
    self.register_hook = False
       
  def forward(self, x):
    conv1_out = self.conv1(x)
    relu1_out = self.relu1(conv1_out)
    conv2_out = self.conv2(relu1_out)
    relu2_out = self.relu2(conv2_out)
    flatten_out = self.flatten(relu2_out)
    fc_out = self.fc(flatten_out)

    if self.register_hook:
        conv1_out.register_hook(lambda grad: self.hook_fn(grad=grad,
            name='conv1_out'))
        relu1_out.register_hook(lambda grad: self.hook_fn(grad=grad,
            name='relu1_out'))
        conv2_out.register_hook(lambda grad: self.hook_fn(grad=grad,
            name='conv2_out'))
        relu2_out.register_hook(lambda grad: self.hook_fn(grad=grad,
            name='relu2_out'))
        fc_out.register_hook(lambda grad: self.hook_fn(grad=grad,
            name='fc_out'))
    
    return fc_out
  
  def hook_fn(self, grad, name):
    self.hook[name].append(grad)

  def reset_hook(self):
    self.hook = {'conv1_out': [],'relu1_out': [],'conv2_out': [],'relu2_out': [],'fc_out': []}

class RNN(nn.Module):
  def __init__(self, config):
    super(RNN, self).__init__()
    self.input_size = config.get("input_size")
    self.num_layers = config.get("num_layers")
    self.hidden_size = config.get("hidden_size")
    self.output_size = config.get("output_size")
    
    self.rnn = nn.RNN(self.input_size, self.hidden_size, self.num_layers, batch_first=True)
    self.fc = nn.Linear(self.hidden_size, self.output_size)
    
    self.hook = {'fc_out': [], 'rnn_out': []}
    self.register_hook = False
      
  def init_hidden(self):
    return (torch.zeros(self.num_layers, self.batch_size, self.hidden_size).to(device))
  
  def forward(self, x):
    self.batch_size = x.size(0)
    self.hidden = self.init_hidden()
    out, self.hidden = self.rnn(x, self.hidden)
    fc_out = self.fc(out)
    
    if self.register_hook:
        fc_out.register_hook(lambda grad: self.hook_fn(grad=grad,
            name='fc_out'))
        out.register_hook(lambda grad: self.hook_fn(grad=grad,
            name='rnn_out'))
        
    return fc_out
  
  def hook_fn(self, grad, name):
    self.hook[name].append(grad)

  def reset_hook(self):
    self.hook = {'fc_out': [], 'rnn_out': []}


def get_data():

  full_dataset = ImageDataLoader()
  
  return full_dataset

def make_loader(dataset, batch_size):

  loader = DataLoader(dataset=dataset, 
                      batch_size=batch_size, 
                      shuffle=False)
  return loader


def main(encoder, model, data_loader, criterion, optimizer):
    model.eval()
    encoder.eval()
    model.register_hook=True
    encoder.register_hook=True
    model.reset_hook()
    encoder.reset_hook()
    mm_image = None
    losses = []
    for idx, (x_t, x_tp1, labels) in enumerate(data_loader): # datloader must be batch_size = 1
        x_t, x_tp1, labels = x_t.to(device), x_tp1.to(device), (labels.float()).to(device)
        # halt the visual flow
        if idx in range(400,500):
            # if mm_image is None:
            #     mm_image = x_t
            # x_t = mm_image
            # x_tp1 = mm_image
            x_tp1 = x_t.clone()

        s_t = encoder(x_t)
        s_tp1 = encoder(x_tp1)

        batch_size1 = len(s_t)
        batch_size2 = len(s_tp1)

        output1 = s_t.reshape(batch_size1,1,-1)
        output2 = s_tp1.reshape(batch_size2,1,-1)
        
        seq = torch.cat((output1, output2.detach()), dim=1)

        outputs = model(seq.to(device))
        
        optimizer.zero_grad()
        loss = criterion(outputs[:,-1].squeeze(), labels.squeeze())
        losses.append(loss.item())
        loss.backward()

def mismatching(encoder, model, optimizer, criterion, data_loader):

    # make the model, data, and optimization problem
    print(encoder)
    print(model)

    main(encoder, model, data_loader, criterion, optimizer)


    return model

PATH = "checkpoints/changingEncoder/3.pt"

encoder = Encoder()
model = RNN(config)
optimizer = torch.optim.Adam(model.parameters(), 
                                lr=config.get("learning_rate"))

checkpoint = torch.load(PATH, map_location=torch.device('cpu'))
encoder.load_state_dict(checkpoint['encoder_state_dict'])
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
# Make the data
dataset = get_data()
data_loader = make_loader(dataset, batch_size=config.get("mismatch_batch_size"))
criterion = nn.MSELoss()

model = mismatching(encoder, model, optimizer, criterion, data_loader)

class MidpointNormalize(colors.Normalize):
    """Normalise the colorbar."""
    def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
        self.midpoint = midpoint
        colors.Normalize.__init__(self, vmin, vmax, clip)
    def __call__(self, value, clip=None):
        x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
        return np.ma.masked_array(np.interp(value, x, y), np.isnan(value))

def plot_heatmap(data, title, save_dir):
    plt.figure(figsize=(6,5))
    ax = plt.subplot(111)
    sorted_data = data[np.mean(data, axis=1).argsort()][::-1]
    print(sorted_data)
    # sorted_data = data
    min_, max_ = -np.max(np.abs(sorted_data)), np.max(np.abs(sorted_data))
    mymin, mymax =min_, max_
    print(mymin, mymax)
    cmap=matplotlib.cm.RdBu_r
    im = ax.imshow(sorted_data, norm=MidpointNormalize(mymin, mymax, 0.),
                    aspect='auto',interpolation='nearest', cmap=cmap)
    ax.tick_params(
            axis='x',          # changes apply to the x-axis
            which='both',      # both major and minor ticks are affected
            bottom=True,      # ticks along the bottom edge are off
            top=False,         # ticks along the top edge are off
            labelbottom=True) # labels along the bottom edge are off
    ax.set_xlabel('Time [s]')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    plt.colorbar(im, cax=cax, orientation='vertical')
    ax.set_ylabel('Sorted neurons #')
    ax.set_yticks(np.arange(0,len(data),5)) 
    plt.suptitle(title)
    plt.savefig(save_dir, dpi=300)

def plot_heatmap_rnn(data, title, save_dir):
    plt.figure(figsize=(6,5))
    ax = plt.subplot(111)
    sorted_data = data[np.mean(data, axis=1).argsort()][::-1]
    # sorted_data = data
    min_, max_ = -np.max(np.abs(sorted_data)), np.max(np.abs(sorted_data))
    mymin, mymax =min_, max_
    print(mymin, mymax)
    cmap=matplotlib.cm.RdBu_r
    im = ax.imshow(sorted_data, norm=MidpointNormalize(mymin, mymax, 0.),
                    aspect='auto',interpolation='nearest', cmap=cmap)
    ax.tick_params(
            axis='y',          # changes apply to the x-axis
            which='both',      # both major and minor ticks are affected
            left=False,
            labelleft=False) # labels along the bottom edge are off
    ax.set_xlabel('Time [s]')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    plt.colorbar(im, cax=cax, orientation='vertical')
    ax.set_ylabel('Sorted neurons #')
    plt.suptitle(title)
    plt.savefig(save_dir, dpi=300)

enc_fc_out = []
rnn_fc_out = []

for i, (elem) in enumerate(encoder.hook['fc_out']):
    if i in range(380,400):
      arr = (elem.flatten()).numpy()
      enc_fc_out.append(arr)
    elif i in range(440,460):
      arr = (elem.flatten()).numpy()
      enc_fc_out.append(arr)
    elif i in range(500,520):
      arr = (elem.flatten()).numpy()
      enc_fc_out.append(arr)

for i, (elem) in enumerate(model.hook['fc_out']):
    if i in range(380,400):
      arr = (elem.flatten()).numpy()
      rnn_fc_out.append(arr)
    elif i in range(440,460):
      arr = (elem.flatten()).numpy()
      rnn_fc_out.append(arr)
    elif i in range(500,520):
      arr = (elem.flatten()).numpy()
      rnn_fc_out.append(arr)

enc_fc_out = np.asarray(enc_fc_out)
rnn_fc_out = np.asarray(rnn_fc_out)

plot_heatmap(-(enc_fc_out.transpose()), "heatmap", "plots/changingEncoder/enc_3_sliced" )
plot_heatmap_rnn(-(rnn_fc_out.transpose()), "heatmap", "plots/changingEncoder/rnn_3_sliced" )


