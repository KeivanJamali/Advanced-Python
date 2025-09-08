import torch.nn as nn

class NN(nn.Module):
    def __init__(self, features_n, output_n, hidden_n):
        super().__init__()
        self.l_0 = nn.Linear(in_features=features_n, out_features=hidden_n)
        self.l_1 = nn.Linear(in_features=hidden_n, out_features=hidden_n)
        self.l_00 = nn.Linear(in_features=hidden_n, out_features=output_n)
        self.relu = nn.ReLU()

    def forward(self, x):
        output = self.l_0(x)
        output = self.relu(output)
        output = self.l_1(output)
        output = self.relu(output)
        output = self.l_00(output)
        return output