""" L1 norm, l2 norm, max norm, l2 squared norm"""
import torch


def l1norm(vector: torch.tensor) -> float:
    return torch.norm(vector, p=1)


def l2norm(vector: list) -> float:
    return torch.norm(vector, p=2)


def l2squarred(vector: list) -> float:
    return torch.norm(vector, p=2) ** 2


def maxnorm(vector: list) -> float:
    return abs(max(vector))


print(l1norm(vector=torch.tensor(data=[1.0, -2.0, 3.0])))
print(l2norm(vector=torch.tensor(data=[1.0, -2.0, 3.0])))
print(l2squarred(vector=torch.tensor(data=[1.0, -2.0, 3.0])))
print(maxnorm(vector=torch.tensor(data=[1.0, -2.0, 3.0])))
