import torch
from torch.distributions.uniform import Uniform


def randint(low, high):
    """
    call torch.randint to preserve random among dataloader workers
    """
    return int(torch.randint(low, high, (1, )))


def rand():
    """
    call torch.rand to preserve random among dataloader workers
    """
    return float(torch.rand((1, )))


def rand_uniform(low, high):
    """
    call torch uniform to preserve random among dataloader workers
    """
    return float(Uniform(low, high).sample())

