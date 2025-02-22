import torch
import torch.nn as nn
# import pytorch_lightning as pl
from .set_transformer import ISAB, PMA

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class SetEncoder(nn.Module):
    def __init__(self, cfg):
        super(SetEncoder, self).__init__()
        self.linear = cfg.linear
        self.bit16 = cfg.bit16
        self.norm = cfg.norm
        assert cfg.linear != cfg.bit16, "one and only one between linear and bit16 must be true at the same time"
        if cfg.norm:
            self.register_buffer("mean", torch.tensor(cfg.mean))
            self.register_buffer("std", torch.tensor(cfg.std))

        self.activation = cfg.activation
        self.input_normalization = cfg.input_normalization
        if cfg.linear:
            self.linearl = nn.Linear(cfg.dim_input, 16 * cfg.dim_input)
        self.selfatt = nn.ModuleList()
        #dim_input = 16*dim_input
        self.selfatt1 = ISAB(16 * cfg.dim_input, cfg.dim_hidden,
                             cfg.num_heads, cfg.num_inds, ln=cfg.ln)
        for i in range(cfg.n_l_enc):
            self.selfatt.append(
                ISAB(cfg.dim_hidden, cfg.dim_hidden, cfg.num_heads, cfg.num_inds, ln=cfg.ln))
        self.outatt = PMA(cfg.dim_hidden, cfg.num_heads,
                          cfg.num_features, ln=cfg.ln)
        self.dim_input = cfg.dim_input

    def float2bit(self, f, num_e_bits=5, num_m_bits=10, bias=127., dtype=torch.float32):
        # SIGN BIT
        # Swap plus and minus => 0 is plus and 1 is minus
        s = (torch.sign(f + 0.001) * -1 + 1) * 0.5
        s = s.unsqueeze(-1)
        f1 = torch.abs(f)
        # EXPONENT BIT
        e_scientific = torch.floor(torch.log2(f1))
        e_scientific[e_scientific == float(
            "-inf")] = -(2**(num_e_bits - 1) - 1)
        e_decimal = e_scientific + (2**(num_e_bits - 1) - 1)
        e = self.integer2bit(e_decimal, num_bits=num_e_bits)
        # MANTISSA
        f2 = f1 / 2**(e_scientific)
        m2 = self.remainder2bit(f2 % 1, num_bits=bias)
        fin_m = m2[:, :, :, :num_m_bits]  # [:,:,:,8:num_m_bits+8]
        return torch.cat([s, e, fin_m], dim=-1).type(dtype)

    def remainder2bit(self, remainder, num_bits=127):
        dtype = remainder.type()
        exponent_bits = torch.arange(num_bits, device=DEVICE).type(dtype)
        exponent_bits = exponent_bits.repeat(remainder.shape + (1,))
        out = (remainder.unsqueeze(-1) * 2 ** exponent_bits) % 1
        return torch.floor(2 * out)

    def integer2bit(self, integer, num_bits=8):
        dtype = integer.type()
        exponent_bits = - \
            torch.arange(-(num_bits - 1), 1, device=DEVICE).type(dtype)
        exponent_bits = exponent_bits.repeat(integer.shape + (1,))
        out = integer.unsqueeze(-1) / 2 ** exponent_bits
        return (out - (out % 1)) % 2

    def forward(self, x):
        # Pad empty xvars in input
        if self.dim_input != x.shape[2]:
            y = x[:,:,-1:]
            xs = x[:,:,:-1]
            padd = torch.zeros(x.shape[0], x.shape[1], self.dim_input - x.shape[2], dtype=x.dtype).to(DEVICE)
            x = torch.cat([xs, padd, y], axis=2)

        if self.bit16:
            x = self.float2bit(x)
            x = x.view(x.shape[0], x.shape[1], -1)  # [13, 152, 64]
            if self.norm:
                x = (x - 0.5) * 2
        if self.input_normalization:
            means = x[:, :, -1].mean(axis=1).reshape(-1, 1)
            std = x[:, :, -1].std(axis=1).reshape(-1, 1)
            std[std == 0] = 1
            x[:, :, -1] = (x[:, :, -1] - means) / std

        if self.linear:
            if self.activation == 'relu':
                x = torch.relu(self.linearl(x))
            elif self.activation == 'sine':
                x = torch.sin(self.linearl(x))
            else:
                x = (self.linearl(x))
        x = self.selfatt1(x)
        for layer in self.selfatt:  # x : [13, 152, 512]
            x = layer(x)
        x = self.outatt(x)
        # breakpoint()
        return x


if __name__ == "__main__":
    model = SetEncoder(2, 2, 6, 2, 3, 1, 3, 1, 'linear', 0, 0, 1, 1, True)
    #self, n_l,dim_input,dim_hidden,num_heads,num_inds,ln,num_features,linear,activation,bit16,norm,mean,std,input_normalization
    print(model)
    model.eval()
    x = torch.Tensor([[1, 2, 3, 4, 5, 6], [7, 8, 9, 10, 11, 12]]
                     ).T.unsqueeze(0).float()
    x1 = torch.Tensor(
        [[6, 3, 2, 4, 5, 1], [12, 9, 8, 10, 11, 7]]).T.unsqueeze(0).float()
    print(x.max())
    print(model(x))
    print(model(x1))
