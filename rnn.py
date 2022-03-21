from numpy import dtype
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torch.utils.data as load

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = 'cpu'

num_epochs = 10
batch_size = 4
learning_rate = 0.005
input_size = 4096
output_size = 1
sequence_length = 2
hidden_size = 128
num_layers = 2

data_transform = transforms.Compose([transforms.Grayscale(num_output_channels=1), transforms.ToTensor(), transforms.Lambda(lambda x: torch.flatten(x))])
dataset = torchvision.datasets.ImageFolder(root='trainingData', transform=data_transform)
train_loader = load.DataLoader(dataset, batch_size=batch_size, shuffle=False)

class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(RNN, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device) 
        out, _ = self.rnn(x, h0)  
        out = out[:, -1]
        out = self.fc(out)
        return out

model = RNN(input_size, hidden_size, num_layers, output_size).to(device)

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)


n_total_steps = len(train_loader)
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        images = images.reshape(-1, sequence_length, input_size).to(device)
        labels = (labels.float()).to(device)
        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if (i+1) % 10 == 0:
            print (f'Epoch [{epoch+1}/{num_epochs}] | Step [{i+1}/{n_total_steps}] | Loss: {loss.item():.4f}')
