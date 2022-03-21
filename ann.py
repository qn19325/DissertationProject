import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import torch.utils.data as load


# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = 'cpu'

batch_size = 4
input_size = 8192
hidden_size = 128
output_size = 1
num_epochs = 100
learning_rate = 0.01

data_transform = transforms.Compose([transforms.Grayscale(num_output_channels=1), transforms.ToTensor(), transforms.Lambda(lambda x: torch.flatten(x))])
dataset = torchvision.datasets.ImageFolder(root='trainingData', transform=data_transform)
train_loader = load.DataLoader(dataset, batch_size=batch_size, shuffle=False)


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

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)


for epoch in range(num_epochs):
    for i, (images,labels) in enumerate(train_loader):
        images = images.reshape(-1, input_size).to(device)
        labels = (labels.float()).to(device)
        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print (f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')