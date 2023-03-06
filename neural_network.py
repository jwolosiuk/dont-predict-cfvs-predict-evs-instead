from torch import nn


class MLP(nn.Module):
    def __init__(self, input_size, target_size, hidden_sizes, dropout):
        super(MLP, self).__init__()
        n_layers = len(hidden_sizes)
        hidden_sizes[:0] = [input_size]
        layers = [
            building_block(hidden_sizes[i], hidden_sizes[i + 1], dropout if i > 0 else 0) for i in range(n_layers)
        ]
        layers.append(nn.Linear(hidden_sizes[-1], target_size))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        x = self.layers(x)
        return x


def building_block(in_f, out_f, dropout):
    if dropout > 0:
        return nn.Sequential(nn.Linear(in_f, out_f), nn.PReLU(), nn.BatchNorm1d(out_f), nn.Dropout(dropout))
    else:
        return nn.Sequential(nn.Linear(in_f, out_f), nn.PReLU(), nn.BatchNorm1d(out_f))
