import torch
import torch.nn as nn

class GravityNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.beta = nn.Parameter(torch.tensor(0.1, dtype=torch.float32))
        self.k = nn.Parameter(torch.tensor(1, dtype=torch.float32))

    def forward(self, production, attraction, cost):
        """
        Compute the gravity flow matrix T_ij.

        productions: (B, N) - batch of origin productions
        attractions: (B, N) - batch of destination attractions
        cost: (B, N, N) - distance between all pairs
        """ 
        print(production)
        F = torch.exp(-self.beta * cost)
        PA = production * attraction 
        Tij = self.k * PA * F 

        return Tij