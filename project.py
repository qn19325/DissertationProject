import torch
import torch.nn as nn

# function = 2x + 3
X = torch.tensor([[0], [1], [2], [3], [4], [5], [6], [7], [8], [9], [10]], dtype=torch.float32)
Y = torch.tensor([[3], [5], [7], [9], [11], [13], [15], [17], [19], [21], [23]], dtype=torch.float32)

samples, features = X.shape
print(samples, features)

input_size = features
output_size = features

class LinearRegression(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LinearRegression, self).__init__()
        self.lin = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.lin(x)

model = LinearRegression(input_size, output_size)

# training
loss = nn.MSELoss()
update = torch.optim.SGD(model.parameters(), lr=0.02)

for epoch in range(50):
    # prediction = forward pass
    y_prediction = model(X)
    print(y_prediction)
    # calculate MSE between prediction and actual value
    l = loss(Y, y_prediction)
    # gradients = backward pass
    l.backward()
    # update weights
    update.step()
    # zero the gradients
    update.zero_grad()
    if epoch % 1 == 0:
        print(f'epoch {epoch}, loss = {l}')