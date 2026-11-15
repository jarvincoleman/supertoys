### imports
import torch
import numpy as np



x = torch.rand(4,5) 
mask = torch.rand(4, 5) > 0.9
print(mask)
print(x* mask)




