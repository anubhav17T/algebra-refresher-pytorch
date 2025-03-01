import torch

vector = torch.tensor(data=[3.0, 4.0])
print(vector.shape)

#magnitude
print(torch.norm(vector,p=2))
