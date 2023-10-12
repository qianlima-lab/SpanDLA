import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, input_size, ouput_size):
        super(MLP, self).__init__()
        self.linear = nn.Sequential(
            nn.Linear(input_size, input_size // 2),
            nn.ReLU(),
            nn.Linear(input_size // 2, input_size // 4),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(input_size // 4, ouput_size)
        )

    def forward(self, x):
        return self.linear(x)
