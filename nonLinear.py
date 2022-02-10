import torch
from torch import nn 
import torch.nn.functional as Func
import torchvision
from torchvision import transforms, datasets
import wandb

wandb.init(project="my-test-project", entity="qn19325")
wandb.config = {
  "learning_rate": 0.001,
  "epochs": 3,
  "batch_size": 10
}


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
loss_func = nn.CrossEntropyLoss()

for epoch in range(3): 
    for data in training_set:  # `data` is a batch of data
        X, y = data  # X i= features, y = targets (for current batch)
        network.zero_grad()  # sets gradients to 0 before calculation of loss
        output = network(X.view(-1,784))  # reshape the batch (28x28 -> 1x784)
        loss = loss_func(output, y)  # calculate the loss
        loss.backward()  # backprop through network
        optimizer.step()  # update weights
    wandb.log({"loss": loss})
    print(f'loss: {loss}')
    