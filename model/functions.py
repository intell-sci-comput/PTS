import torch
import torch.nn as nn
import torch.nn.functional as F

from model.operators import (Identity_op,
                             Sin_op,
                             Cos_op,
                             Exp_op,
                             Log_op,
                             Neg_op,
                             Inv_op,
                             Add_op,
                             Mul_op,
                             Div_op,
                             Sub_op,
                             SemiDiv_op,
                             SemiSub_op)

from model.operators import (Sign_op,
                             Pow2_op,
                             Pow3_op,
                             Pow_op,
                             Sigmoid_op,
                             Abs_op,
                             Cosh_op,
                             Tanh_op,
                             Sqrt_op)


class Identity(nn.Module):
    def __init__(self, in_dim, device):
        super(Identity, self).__init__()

        self.in_dim = in_dim
        self.out_dim = in_dim
        self.is_unary = True
        self.is_directed = True

        self.operator = Identity_op()

    def forward(self, x):
        return x

# sin

class Sin(nn.Module):
    def __init__(self, in_dim, device):
        super(Sin, self).__init__()

        self.in_dim = in_dim
        self.out_dim = in_dim
        self.is_unary = True
        self.is_directed = True

        self.operator = Sin_op()

    def forward(self, x):
        return torch.sin(x)

# cos


class Cos(nn.Module):
    def __init__(self, in_dim, device):
        super(Cos, self).__init__()

        self.in_dim = in_dim
        self.out_dim = in_dim
        self.is_unary = True
        self.is_directed = True

        self.operator = Cos_op()

    def forward(self, x):
        return torch.cos(x)


class Exp(nn.Module):
    def __init__(self, in_dim, device, threshold=10):
        super(Exp, self).__init__()
        self.threshold = threshold

        self.in_dim = in_dim
        self.out_dim = in_dim
        self.is_unary = True
        self.is_directed = True

        self.operator = Exp_op()

    def forward(self, x):
        return torch.exp(x)


class Log(nn.Module):
    def __init__(self, in_dim, device, threshold=1e-10):
        super(Log, self).__init__()
        self.threshold = threshold

        self.in_dim = in_dim
        self.out_dim = in_dim
        self.is_unary = True
        self.is_directed = True

        self.operator = Log_op()

    def forward(self, x):

        log = torch.log(x)
        return log

# neg

class Neg(nn.Module):
    def __init__(self, in_dim, device):
        super(Neg, self).__init__()

        self.in_dim = in_dim
        self.out_dim = in_dim
        self.is_unary = True
        self.is_directed = True

        self.operator = Neg_op()

    def forward(self, x):
        return -x

class Inv(nn.Module):
    def __init__(self, in_dim, device, threshold=1e-10):
        super(Inv, self).__init__()
        self.threshold = threshold

        self.in_dim = in_dim
        self.out_dim = in_dim
        self.is_unary = True
        self.is_directed = True

        self.operator = Inv_op()

    def forward(self, x):
        return 1 / x

class Mul(nn.Module):
    def __init__(self, in_dim, device):
        super(Mul, self).__init__()

        self.in_dim = in_dim
        self.out_dim = in_dim * (in_dim + 1) // 2
        self.is_unary = False
        self.is_directed = False
        self.operator = Mul_op()
        self.device = device

    def forward(self, x):
        indices = torch.triu_indices(
            self.in_dim, self.in_dim, offset=0, dtype=torch.int32, device=x.device
        )
        out = x[:, indices[0]] * x[:, indices[1]]
        return out


class Add(nn.Module):
    def __init__(self, in_dim, device):
        super(Add, self).__init__()

        self.in_dim = in_dim
        self.out_dim = in_dim * (in_dim + 1) // 2
        self.is_unary = False
        self.is_directed = False

        self.operator = Add_op()
        self.device = device

    def forward(self, x):
        indices = torch.triu_indices(
            self.in_dim, self.in_dim, offset=0, dtype=torch.int32, device=x.device
        )
        out = x[:, indices[0]] + x[:, indices[1]]
        return out


class Div(nn.Module):
    def __init__(self, in_dim, device, threshold=1e-10):
        super(Div, self).__init__()

        self.threshold = threshold

        self.in_dim = in_dim
        self.out_dim = in_dim * in_dim
        self.is_unary = False
        self.is_directed = True

        self.operator = Div_op()

        
    def forward(self, x):
    
        num = x.view(1, -1, 1)
        deno = x.view(1, 1, -1)
        out = (num / deno).view(1, -1)
        
        return out

class Sub(nn.Module):
    def __init__(self, in_dim, device):
        super(Sub, self).__init__()

        self.in_dim = in_dim
        self.out_dim = in_dim * in_dim
        self.is_unary = False
        self.is_directed = True

        self.operator = Sub_op()

    def forward(self, x):
        num = x.view(1, -1, 1)
        deno = x.view(1, 1, -1)
        out = (num - deno).view(1, -1)
        return out



class SemiDiv(nn.Module):
    def __init__(self, in_dim, device, threshold=1e-10):
        super(SemiDiv, self).__init__()

        self.threshold = threshold

        self.in_dim = in_dim
        self.out_dim = in_dim * (in_dim + 1) // 2
        self.is_unary = False
        self.is_directed = False

        self.operator = SemiDiv_op()
        self.device = device

    def forward(self, x):
        indices = torch.triu_indices(
            self.in_dim, self.in_dim, offset=0, dtype=torch.int32, device=x.device
        )
        deno = x[:, indices[1]]
        num = x[:, indices[0]]
        out = num / deno
        return out


class SemiSub(nn.Module):
    def __init__(self, in_dim, device):
        super(SemiSub, self).__init__()

        self.in_dim = in_dim
        self.out_dim = in_dim * (in_dim + 1) // 2
        self.is_unary = False
        self.is_directed = False

        self.operator = SemiSub_op()
        self.device = device

    def forward(self, x):
        # x (bs, dim)
        indices = torch.triu_indices(
            self.in_dim, self.in_dim, offset=0, dtype=torch.int32, device=x.device
        )
        out = (x[:, indices[0]] - x[:, indices[1]])
        return out

class Sign(nn.Module):
    def __init__(self, in_dim, device):
        super(Sign, self).__init__()

        self.in_dim = in_dim
        self.out_dim = in_dim
        self.is_unary = True
        self.is_directed = True

        self.operator = Sign_op()

    def forward(self, x):
        return torch.sign(x)


class Pow2(nn.Module):
    def __init__(self, in_dim, device):
        super(Pow2, self).__init__()

        self.in_dim = in_dim
        self.out_dim = in_dim
        self.is_unary = True
        self.is_directed = True

        self.operator = Pow2_op()

    def forward(self, x):
        return x ** 2


class Pow3(nn.Module):
    def __init__(self, in_dim, device):
        super(Pow3, self).__init__()

        self.in_dim = in_dim
        self.out_dim = in_dim
        self.is_unary = True
        self.is_directed = True

        self.operator = Pow3_op()

    def forward(self, x):
        return x ** 3

class Pow(nn.Module):
    def __init__(self, in_dim, device):
        super(Pow, self).__init__()

        self.in_dim = in_dim
        self.out_dim = in_dim * in_dim
        self.is_unary = False
        self.is_directed = True

        self.operator = Pow_op()

    def forward(self, x):
        # x (bs, dim)
        deno = x.reshape(1, 1, -1)
        num = x.reshape(1, -1, 1)
        out = num ** deno
        out = out.reshape(1, -1)
        return out


class Sigmoid(nn.Module):
    def __init__(self, in_dim, device):
        super(Sigmoid, self).__init__()

        self.in_dim = in_dim
        self.out_dim = in_dim
        self.is_unary = True
        self.is_directed = True

        self.operator = Sigmoid_op()

    def forward(self, x):
        return 1/(1 + torch.exp(-x))

class Abs(nn.Module):
    def __init__(self, in_dim, device):
        super(Abs, self).__init__()

        self.in_dim = in_dim
        self.out_dim = in_dim
        self.is_unary = True
        self.is_directed = True

        self.operator = Abs_op()

    def forward(self, x):
        return torch.abs(x)

class Cosh(nn.Module):
    def __init__(self, in_dim, device):
        super(Cosh, self).__init__()

        self.in_dim = in_dim
        self.out_dim = in_dim
        self.is_unary = True
        self.is_directed = True

        self.operator = Cosh_op()

    def forward(self, x):
        return torch.cosh(x)

class Tanh(nn.Module):
    def __init__(self, in_dim, device):
        super(Tanh, self).__init__()

        self.in_dim = in_dim
        self.out_dim = in_dim
        self.is_unary = True
        self.is_directed = True

        self.operator = Tanh_op()

    def forward(self, x):
        return torch.tanh(x)

class Sqrt(nn.Module):
    def __init__(self, in_dim, device):
        super(Sqrt, self).__init__()

        self.in_dim = in_dim
        self.out_dim = in_dim
        self.is_unary = True
        self.is_directed = True

        self.operator = Sqrt_op()

    def forward(self, x):
        return torch.sqrt(x)

if __name__ == '__main__':

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    x = torch.randint(-1, 5, (3, 5))
    print(x)
    layer = Mul(5, device)
    out = layer(x)
    print(out.shape)
    print(out)
