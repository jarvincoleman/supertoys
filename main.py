### imports
import torch
import numpy as np





def batch(batch_size=1024, features=5, sparsity=0.9): 
    values = torch.rand(batch_size, features)
    is_active = torch.rand(batch_size, features) > sparsity
    return values * is_active

features = 5 
hidden_dimensions = 2

weights = torch.nn.Parameter(torch.randn(features, hidden_dimensions) * 0.1)
bias = torch.nn.Parameter(torch.zeros(features))

def forward(inputs): 
    hidden = inputs @ weights
    outputs = hidden @ weights.T + bias
    return torch.relu(outputs) 

inputs = batch(4, 5) 
print(forward(inputs))

optimizer = torch.optim.Adam([weights, bias], lr=1e-3)

for step in range(5000): 
    inputs = batch(1024) 
    outputs = forward(inputs) 
    loss = ((outputs - inputs) ** 2).mean()

    optimizer.zero_grad() 
    loss.backward() 
    optimizer.step() 

    if step % 500 == 0:
        print(step, loss.item())
    

# x = torch.rand(4,5) 
# mask = torch.rand(4, 5) > 0.9
# print(mask)
# print(x* mask)