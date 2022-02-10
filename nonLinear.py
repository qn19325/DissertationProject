import torch
from torch import nn 
import torch.nn.functional as Func
import torchvision
from torchvision import transforms, datasets

training = datasets.MNIST("", train=True, download=True, transform=transforms.Compose([transforms.ToTensor()]))
training_set = torch.utils.data.DataLoader(training, batch_size=10, shuffle=True)

class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(28*28, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 10)
    
    def forward(self, x):
        x = Func.relu(self.fc1(x))
        x = Func.relu(self.fc2(x))
        output = self.fc3(x)
        return Func.log_softmax(output, dim=1)

network = NeuralNetwork()

optimizer = torch.optim.SGD(network.parameters(), lr=0.001)
loss = nn.CrossEntropyLoss()

for epoch in range(3): 
    for data in training_set:  # `data` is a batch of data
        X, y = data  # X i= features, y = targets (for current batch)
        network.zero_grad()  # sets gradients to 0 before calculation of loss
        output = network(X.view(-1,784))  # reshape the batch (28x28 -> 1x784)
        l = loss(output, y)  # calculate the loss
        l.backward()  # backprop through network
        optimizer.step()  # update weights
    print(f'loss: {l}')
    