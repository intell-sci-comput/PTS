from torch import nn, Tensor
import torch


class DummyTokenMLPEncoder(nn.Module):
    # Only works for fixed input dim inputs
    def __init__(self, input_dim, output_dim, HIDDEN_UNITS=64):
        super(DummyTokenMLPEncoder, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.linear_tanh_stack = nn.Sequential(
            nn.Linear(input_dim, HIDDEN_UNITS),
            nn.Tanh(),
            nn.Linear(HIDDEN_UNITS, HIDDEN_UNITS),
            nn.Tanh(),
            nn.Linear(HIDDEN_UNITS, output_dim),
        )

        for m in self.linear_tanh_stack.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)

    def forward(self, i):
        return torch.mean(self.linear_tanh_stack(i.view(-1, self.input_dim)).view(i.shape[0], i.shape[1], self.output_dim), axis=1)


class DummyTokenGRUEncoder(nn.Module):
    def __init__(self, dimension_in, latents, hidden_units):
        super(DummyTokenGRUEncoder, self).__init__()
        self.gru = nn.GRU(
            dimension_in, hidden_units, 2, batch_first=True)
        self.linear_out = nn.Linear(hidden_units, latents)
        nn.init.xavier_uniform_(self.linear_out.weight)

    def forward(self, i):
        out, _ = self.gru(i)
        return self.linear_out(out[:, -1, :])


if __name__ == '__main__':
    tokens = torch.rand(100, 123)
    tokens = tokens.unsqueeze(-1)
    gru = DummyTokenGRUEncoder(1, 32, 64)
    out = gru(tokens)
    assert out.shape[0] == 100
    assert out.shape[1] == 32

    data_to_encode = torch.rand(4000, 20, 2)
    encoder = DummyTokenMLPEncoder(2, 32)
    out = encoder(data_to_encode)
    assert out.shape[0] == data_to_encode.shape[0]
    assert out.shape[1] == 32
