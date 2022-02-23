####### TOO MANY EPOCHS TO RUN ON WANDB QUICKLY #######

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import wandb

wandb.init(project="nonLinear", entity="qn19325")
wandb.config = {
  "learning_rate": 0.001,
  "epochs": 20000,
  "batch_size": 11
}

x = torch.tensor([[0], [1], [2], [3], [4], [5], [6], [7], [8], [9], [10]], dtype=torch.float32)
y = x**2 + 4

class Network(torch.nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Network, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)   # hidden layer
        self.fc2 = nn.Linear(hidden_size, output_size)   # output layer

    def forward(self, x):
        x = F.relu(self.fc1(x))      # activation function
        x = self.fc2(x)             # linear output
        return x

model = Network(input_size=1, hidden_size=10, output_size=1)     # define the network
# print(net)  # net architecture
optimizer = optim.SGD(model.parameters(), lr=0.001)
loss_func = nn.MSELoss()  # mean squared error loss

# train the network
for epoch in range(20000):
  
    prediction = model(x)     # predict based on input x
    loss = loss_func(prediction, y)

    optimizer.zero_grad()   # zero gradients
    loss.backward()         # backpropagation through network
    optimizer.step()        # update gradients
    if epoch % 10 == 0:
        print(prediction)
        print(f'loss: {loss}')
    wandb.log({"loss": loss})
    wandb.watch(model)