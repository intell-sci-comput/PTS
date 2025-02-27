import torch
from torch import nn, optim


class Model(nn.Module):
    def __init__(self, state_size):
        super(Model, self).__init__()

        self.layers = nn.Sequential(nn.Linear(state_size, 256),
                                    nn.Tanh(),
                                    nn.Linear(256, 64),
                                    nn.Tanh(),
                                    nn.Linear(64, 1),
                                    nn.Sigmoid())

    def forward(self, x):
        return self.layers(x)


class ValueFunc:
    def __init__(self, state_size, device):
        self.model = Model(state_size).to(device)
        self.device = device

        self.optimizer = optim.Adam(self.model.parameters())
        self.loss_fn = nn.MSELoss()

    def train(self, states, value):
        """
        Args:
            states: seq len * hidden state size
            value: a scalar, the final value
        """
        self.optimizer.zero_grad()

        values = (torch.ones(states.size(0), 1) * value).to(self.device)
        pred_values = self.model(states)

        loss = self.loss_fn(pred_values, values)

        loss.backward()
        self.optimizer.step()

    def __call__(self, state):
        with torch.no_grad():
            return self.model(state)

    def save_model(self, filename):
        torch.save({self.model.state_dict()}, filename)