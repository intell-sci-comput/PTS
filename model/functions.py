import torch
import torch.nn as nn
import torch.nn.functional as F

from .operators import (
    Identity_op,
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
    SemiSub_op,
)

from .operators import (
    Sign_op,
    Pow2_op,
    Pow3_op,
    Pow_op,
    Sigmoid_op,
    Abs_op,
    Cosh_op,
    Tanh_op,
    Sqrt_op,
)


from abc import abstractmethod


def compute_lcm(A, B):
    # Ensure A and B are tensors with the correct shape
    assert A.shape == B.shape and A.dim() == 2 and A.size(0) == 1

    # Compute conditions
    cond1 = A % B == 0
    cond2 = B % A == 0

    # Initialize C with an empty tensor
    C = torch.empty_like(A)

    # Compute LCM using torch.div with rounding_mode='trunc'
    torch.div(A * B, torch.gcd(A, B), out=C, rounding_mode="trunc")

    # Apply conditions to select the largest element where conditions are met
    torch.where(cond1 | cond2, A, C, out=C)
    torch.where(cond2, B, C, out=C)

    return C


def compute_lcm_batched(A, B, batch_size):
    # Ensure A and B are tensors with the correct shape
    assert A.shape == B.shape and A.dim() == 2

    # Get the total number of elements in A and B
    total_elements = A.shape[1]

    # Initialize C with zeros
    C = torch.zeros_like(A)
    from tqdm import tqdm

    # Process the data in batches
    for start in tqdm(range(0, total_elements, batch_size)):
        end = min(start + batch_size, total_elements)
        batch_A = A[:, start:end]
        batch_B = B[:, start:end]

        # Compute conditions for the current batch
        cond1 = batch_A % batch_B == 0
        cond2 = batch_B % batch_A == 0

        # Compute LCM only for elements where conditions are met
        lcm_cond1 = torch.div(batch_A, batch_B, rounding_mode="trunc") * batch_B
        lcm_cond2 = torch.div(batch_B, batch_A, rounding_mode="trunc") * batch_A
        C[:, start:end][cond1] = lcm_cond1[cond1]
        C[:, start:end][cond2] = lcm_cond2[cond2]

        # Apply conditions to select the largest element where conditions are met
        C[:, start:end][cond1 | cond2] = torch.maximum(batch_A, batch_B)[cond1 | cond2]

        # Clear the GPU cache between batches
        torch.cuda.empty_cache()

    return C


def compute_lcm_cartesian(x):
    # Ensure x is a 1D tensor
    # assert x.dim() == 1

    # Compute the GCD using x and its transpose
    GCD = torch.gcd(x.reshape(1, -1), x.reshape(-1, 1))

    # Compute the LCM
    LCM = (x.reshape(1, -1) * x.reshape(-1, 1)) // GCD

    # Compute conditions
    cond1 = x.reshape(1, -1) % x.reshape(-1, 1) == 0
    cond2 = x.reshape(-1, 1) % x.reshape(1, -1) == 0

    # Apply conditions to select the largest element where conditions are met
    LCM = torch.where(
        cond1 | cond2, torch.maximum(x.reshape(1, -1), x.reshape(-1, 1)), LCM
    )

    # Reshape LCM to 2D tensor
    LCM = LCM.reshape(1, -1)

    return LCM


class CanCountLeaveOperator(nn.Module):
    def count_leave(self, x_leaves):
        if self.is_unary:
            return x_leaves
        elif not self.is_directed:
            in_dim = x_leaves.shape[1]
            indices = torch.triu_indices(
                in_dim, in_dim, offset=0, dtype=torch.int32, device=self.device
            )
            out = x_leaves[:, indices[0]] + x_leaves[:, indices[1]]
            # return out
            return out + 1
        else:
            deno = x_leaves
            num = x_leaves
            deno = deno.reshape(1, 1, -1)
            num = num.reshape(1, -1, 1)
            out = num + deno
            out = out.reshape(1, -1)
            # return out
            return out + 1

    def count_complexity(self, x_cplx):
        if self.is_unary:
            return x_cplx + self.complexity
        elif not self.is_directed:
            in_dim = x_cplx.shape[1]
            indices = torch.triu_indices(
                in_dim, in_dim, offset=0, dtype=torch.int32, device=self.device
            )
            out = x_cplx[:, indices[0]] + x_cplx[:, indices[1]]
            return out + self.complexity
        else:
            deno = x_cplx
            num = x_cplx
            deno = deno.reshape(1, 1, -1)
            num = num.reshape(1, -1, 1)
            out = num + deno
            out = out.reshape(1, -1)
            return out

    def count_prime(self, x_prime):
        if self.is_unary:
            return x_prime
        elif not self.is_directed:
            in_dim = x_prime.shape[1]
            indices = torch.triu_indices(
                in_dim, in_dim, offset=0, dtype=torch.int32, device=self.device
            )
            a = x_prime[:, indices[0]]
            b = x_prime[:, indices[1]]
            # import gc
            # gc.collect()
            # torch.cuda.empty_cache()
            # gc.collect()
            # torch.cuda.empty_cache()
            # return compute_lcm(a, b)
            return compute_lcm_batched(a, b, batch_size=100000)
        else:
            result = compute_lcm_cartesian(x_prime)
            return result


class Identity(CanCountLeaveOperator):
    def __init__(self, in_dim, device):
        super(Identity, self).__init__()

        self.in_dim = in_dim
        self.out_dim = in_dim
        self.is_unary = True
        self.is_directed = True

        self.operator = Identity_op()

        self.complexity = 0

    def forward(self, x, second_device=None):

        if second_device is not None:
            x = x.to(second_device)
        return x


# sin


class Sin(CanCountLeaveOperator):
    def __init__(self, in_dim, device):
        super(Sin, self).__init__()

        self.in_dim = in_dim
        self.out_dim = in_dim
        self.is_unary = True
        self.is_directed = True

        self.operator = Sin_op()

        self.complexity = 4

    def forward(self, x, second_device=None):

        if second_device is not None:
            x = x.to(second_device)
        return torch.sin(x)


# cos


class Cos(CanCountLeaveOperator):
    def __init__(self, in_dim, device):
        super(Cos, self).__init__()

        self.in_dim = in_dim
        self.out_dim = in_dim
        self.is_unary = True
        self.is_directed = True

        self.operator = Cos_op()

        self.complexity = 4

    def forward(self, x, second_device=None):

        if second_device is not None:
            x = x.to(second_device)
        return torch.cos(x)


class Exp(CanCountLeaveOperator):
    def __init__(self, in_dim, device, threshold=10):
        super(Exp, self).__init__()
        self.threshold = threshold

        self.in_dim = in_dim
        self.out_dim = in_dim
        self.is_unary = True
        self.is_directed = True

        self.operator = Exp_op()

        self.complexity = 4

    def forward(self, x, second_device=None):

        if second_device is not None:
            x = x.to(second_device)
        # return torch.exp(torch.clamp(x, max=self.threshold))
        return torch.exp(x)


class Log(CanCountLeaveOperator):
    def __init__(self, in_dim, device, threshold=1e-10):
        super(Log, self).__init__()
        self.threshold = threshold

        self.in_dim = in_dim
        self.out_dim = in_dim
        self.is_unary = True
        self.is_directed = True

        self.complexity = 4

        self.operator = Log_op()

    def forward(self, x, second_device=None):

        if second_device is not None:
            x = x.to(second_device)
        # abs_ = torch.abs(x)
        # clamp = torch.clamp(abs_, min=self.threshold)
        # log = torch.log(clamp)
        log = torch.log(x)
        return log


# neg


class Neg(CanCountLeaveOperator):
    def __init__(self, in_dim, device):
        super(Neg, self).__init__()

        self.in_dim = in_dim
        self.out_dim = in_dim
        self.is_unary = True
        self.is_directed = True

        self.complexity = 1

        self.operator = Neg_op()

    def forward(self, x, second_device=None):

        if second_device is not None:
            x = x.to(second_device)
        return -x


class Inv(CanCountLeaveOperator):
    def __init__(self, in_dim, device, threshold=1e-10):
        super(Inv, self).__init__()
        self.threshold = threshold

        self.in_dim = in_dim
        self.out_dim = in_dim
        self.is_unary = True
        self.is_directed = True

        self.complexity = 1

        self.operator = Inv_op()

    def forward(self, x, second_device=None):

        if second_device is not None:
            x = x.to(second_device)
        # x = torch.where(x < 0, torch.clamp(x, max=-self.threshold), torch.clamp(x, min=self.threshold))
        return 1 / x


class Mul(CanCountLeaveOperator):
    def __init__(self, in_dim, device):
        super(Mul, self).__init__()

        self.in_dim = in_dim
        self.out_dim = in_dim * (in_dim + 1) // 2
        self.is_unary = False
        self.is_directed = False

        self.complexity = 1

        self.operator = Mul_op()
        self.device = device

    def forward(self, x, second_device=None):
        # Recompute indices on-the-fly using torch.int32

        if second_device is not None:
            x = x.to(second_device)

        indices = torch.triu_indices(
            self.in_dim, self.in_dim, offset=0, dtype=torch.int32, device=x.device
        )
        out = x[:, indices[0]] * x[:, indices[1]]
        return out


class Add(CanCountLeaveOperator):
    def __init__(self, in_dim, device):
        super(Add, self).__init__()

        self.in_dim = in_dim
        self.out_dim = in_dim * (in_dim + 1) // 2
        self.is_unary = False
        self.is_directed = False

        self.complexity = 1

        self.operator = Add_op()
        self.device = device

    def forward(self, x, second_device=None):
        # Recompute indices on-the-fly using torch.int32

        # print('x shape',x.shape)

        if second_device is not None:
            x = x.to(second_device)

        indices = torch.triu_indices(
            self.in_dim, self.in_dim, offset=0, dtype=torch.int32, device=x.device
        )
        out = x[:, indices[0]] + x[:, indices[1]]
        return out


class Div(CanCountLeaveOperator):
    def __init__(self, in_dim, device, threshold=1e-10):
        super(Div, self).__init__()

        self.threshold = threshold

        self.in_dim = in_dim
        self.out_dim = in_dim * in_dim
        self.is_unary = False
        self.is_directed = True

        self.complexity = 2

        self.operator = Div_op()

    # def forward(self, x):
    #     deno = x
    #     num = x
    #     deno = deno.reshape(1, 1, -1)
    #     num = num.reshape(1, -1, 1)
    #     out = num / deno
    #     out = out.reshape(1, -1)
    #     return out

    def forward(self, x, second_device=None):

        if second_device is not None:
            x = x.to(second_device)

        num = x.view(1, -1, 1)
        deno = x.view(1, 1, -1)
        out = (num / deno).view(1, -1)

        return out


class Sub(CanCountLeaveOperator):
    def __init__(self, in_dim, device):
        super(Sub, self).__init__()

        self.in_dim = in_dim
        self.out_dim = in_dim * in_dim
        self.is_unary = False
        self.is_directed = True

        self.complexity = 1

        self.operator = Sub_op()

    def forward(self, x, second_device=None):

        if second_device is not None:
            x = x.to(second_device)
        # x (bs, dim)
        # deno = x.reshape(1, 1, -1)
        # num = x.reshape(1, -1, 1)
        # out = num - deno
        # out = out.reshape(1, -1)
        num = x.view(1, -1, 1)
        deno = x.view(1, 1, -1)
        out = (num - deno).view(1, -1)
        return out


# class Mul(CanCountLeaveOperator):
#     def __init__(self, in_dim, device):
#         super(Mul, self).__init__()

#         self.in_dim = in_dim
#         self.out_dim = in_dim * (in_dim + 1) // 2
#         self.is_unary = False
#         self.is_directed = False

#         self.operator = Mul_op()
#         self.device = device

#     def forward(self, x):
#         # Recompute indices on-the-fly using torch.int32
#         indices = torch.triu_indices(
#             self.in_dim, self.in_dim, offset=0, dtype=torch.int32, device=self.device
#         )
#         out = x[:, indices[0]] * x[:, indices[1]]
#         return out


class SemiDiv(CanCountLeaveOperator):
    def __init__(self, in_dim, device, threshold=1e-10):
        super(SemiDiv, self).__init__()

        self.threshold = threshold

        self.in_dim = in_dim
        self.out_dim = in_dim * (in_dim + 1) // 2
        self.is_unary = False
        self.is_directed = False

        self.operator = SemiDiv_op()
        self.device = device

        self.complexity = 2

    def forward(self, x, second_device=None):

        if second_device is not None:
            x = x.to(second_device)
        # x (bs, dim)
        # out = (x[:, self.indices[0]] / x[:, self.indices[1]])
        indices = torch.triu_indices(
            self.in_dim, self.in_dim, offset=0, dtype=torch.int32, device=x.device
        )

        deno = x[:, indices[1]]
        # deno = torch.where(deno < 0, torch.clamp(deno, max=-self.threshold), torch.clamp(deno, min=self.threshold))
        num = x[:, indices[0]]
        out = num / deno
        return out


class SemiSub(CanCountLeaveOperator):
    def __init__(self, in_dim, device):
        super(SemiSub, self).__init__()

        self.in_dim = in_dim
        self.out_dim = in_dim * (in_dim + 1) // 2
        self.is_unary = False
        self.is_directed = False

        self.complexity = 1

        self.operator = SemiSub_op()
        self.device = device

    def forward(self, x, second_device=None):

        if second_device is not None:
            x = x.to(second_device)
        # x (bs, dim)
        indices = torch.triu_indices(
            self.in_dim, self.in_dim, offset=0, dtype=torch.int32, device=x.device
        )
        out = x[:, indices[0]] - x[:, indices[1]]
        return out


class Sign(CanCountLeaveOperator):
    def __init__(self, in_dim, device):
        super(Sign, self).__init__()

        self.in_dim = in_dim
        self.out_dim = in_dim
        self.is_unary = True
        self.is_directed = True

        self.complexity = 4

        self.operator = Sign_op()

    def forward(self, x, second_device=None):

        if second_device is not None:
            x = x.to(second_device)
        return torch.sign(x)


class Pow2(CanCountLeaveOperator):
    def __init__(self, in_dim, device):
        super(Pow2, self).__init__()

        self.in_dim = in_dim
        self.out_dim = in_dim
        self.is_unary = True
        self.is_directed = True

        self.complexity = 2

        self.operator = Pow2_op()

    def forward(self, x, second_device=None):

        if second_device is not None:
            x = x.to(second_device)
        return x**2


class Pow3(CanCountLeaveOperator):
    def __init__(self, in_dim, device):
        super(Pow3, self).__init__()

        self.in_dim = in_dim
        self.out_dim = in_dim
        self.is_unary = True
        self.is_directed = True

        self.complexity = 3

        self.operator = Pow3_op()

    def forward(self, x, second_device=None):

        if second_device is not None:
            x = x.to(second_device)
        return x**3


class Pow(CanCountLeaveOperator):
    def __init__(self, in_dim, device):
        super(Pow, self).__init__()

        self.in_dim = in_dim
        self.out_dim = in_dim * in_dim
        self.is_unary = False
        self.is_directed = True

        self.complexity = 4

        self.operator = Pow_op()

    def forward(self, x, second_device=None):

        if second_device is not None:
            x = x.to(second_device)
        # x (bs, dim)
        deno = x.reshape(1, 1, -1)
        num = x.reshape(1, -1, 1)
        out = num**deno
        out = out.reshape(1, -1)
        return out


class Sigmoid(CanCountLeaveOperator):
    def __init__(self, in_dim, device):
        super(Sigmoid, self).__init__()

        self.in_dim = in_dim
        self.out_dim = in_dim
        self.is_unary = True
        self.is_directed = True

        self.complexity = 6

        self.operator = Sigmoid_op()

    def forward(self, x, second_device=None):

        if second_device is not None:
            x = x.to(second_device)
        return 1 / (1 + torch.exp(-x))


class Abs(CanCountLeaveOperator):
    def __init__(self, in_dim, device):
        super(Abs, self).__init__()

        self.in_dim = in_dim
        self.out_dim = in_dim
        self.is_unary = True
        self.is_directed = True

        self.complexity = 4

        self.operator = Abs_op()

    def forward(self, x, second_device=None):

        if second_device is not None:
            x = x.to(second_device)
        return torch.abs(x)


class Cosh(CanCountLeaveOperator):
    def __init__(self, in_dim, device):
        super(Cosh, self).__init__()

        self.in_dim = in_dim
        self.out_dim = in_dim
        self.is_unary = True
        self.is_directed = True

        self.complexity = 4

        self.operator = Cosh_op()

    def forward(self, x, second_device=None):

        if second_device is not None:
            x = x.to(second_device)
        return torch.cosh(x)


class Tanh(CanCountLeaveOperator):
    def __init__(self, in_dim, device):
        super(Tanh, self).__init__()

        self.in_dim = in_dim
        self.out_dim = in_dim
        self.is_unary = True
        self.is_directed = True

        self.complexity = 4

        self.operator = Tanh_op()

    def forward(self, x, second_device=None):

        if second_device is not None:
            x = x.to(second_device)
        return torch.tanh(x)


class Sqrt(CanCountLeaveOperator):
    def __init__(self, in_dim, device):
        super(Sqrt, self).__init__()

        self.in_dim = in_dim
        self.out_dim = in_dim
        self.is_unary = True
        self.is_directed = True

        self.complexity = 3

        self.operator = Sqrt_op()

    def forward(self, x, second_device=None):

        if second_device is not None:
            x = x.to(second_device)
        return torch.sqrt(x)


if __name__ == "__main__":

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    x = torch.randint(-1, 5, (3, 5))
    print(x)
    layer = Mul(5, device)
    out = layer(x)
    print(out.shape)
    print(out)
