import torch
import torch.nn as nn


class SIFA_MLP(nn.Module):
    def __init__(self, emb_size):
        super().__init__()
        self.linear1 = nn.Linear(emb_size, 2*emb_size)
        self.linear2 = nn.Linear(2*emb_size, emb_size)
        self.activations = [nn.ReLU(), nn.LeakyReLU(), nn.ELU(), nn.GELU()]

    def forward(self, x):
        x = self.linear1(x)
        output = torch.zeros(x.shape).to(x.device)
        for activation in self.activations:
            output += activation(x)
        output = self.linear2(output)
        return output
