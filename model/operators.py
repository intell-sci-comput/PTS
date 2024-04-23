import torch
import torch.nn as nn
import torch.nn.functional as F

# unary operators



class Identity_op():
    def __init__(self):
        super(Identity_op, self).__init__()
        self.is_unary = True
        

    def get_expr(self, sub_expr):
        return sub_expr

    def transform_inputs(self, x):
        return x


class Sin_op():
    def __init__(self):
        super(Sin_op, self).__init__()
        self.is_unary = True

    def get_expr(self, sub_expr):
        return 'sin({})'.format(sub_expr)

    def transform_inputs(self, x):
        return torch.sin(x)


class Cos_op():
    def __init__(self):
        super(Cos_op, self).__init__()
        self.is_unary = True

    def get_expr(self, sub_expr):
        return 'cos({})'.format(sub_expr)

    def transform_inputs(self, x):
        return torch.cos(x)


class Exp_op():
    def __init__(self, threshold=10):
        super(Exp_op, self).__init__()
        self.is_unary = True

        self.threshold = threshold

    def get_expr(self, sub_expr):
        return 'exp({})'.format(sub_expr)

    def transform_inputs(self, x):
        return torch.exp(torch.clamp(x, max=self.threshold))


class Log_op():
    def __init__(self, threshold=1e-10):
        super(Log_op, self).__init__()
        self.is_unary = True

        self.threshold = threshold

    def get_expr(self, sub_expr):
        return 'log(Abs({}))'.format(sub_expr)

    def transform_inputs(self, x):
        abs_ = torch.abs(x)
        clamp = torch.clamp(abs_, min=self.threshold)
        log = torch.log(clamp)
        return log


class Neg_op():
    def __init__(self):
        super(Neg_op, self).__init__()
        self.is_unary = True

    def get_expr(self, sub_expr):
        return '(-({}))'.format(sub_expr)

    def transform_inputs(self, x):
        return -x


class Inv_op():
    def __init__(self, threshold=1e-10):
        super(Inv_op, self).__init__()
        self.is_unary = True

        self.threshold = threshold

    def get_expr(self, sub_expr):
        return '(1/({}))'.format(sub_expr)

    def transform_inputs(self, x):
        x = torch.where(x < 0, torch.clamp(x, max=-self.threshold), torch.clamp(x, min=self.threshold))
        return 1 / x


# binary operators


class Mul_op():
    def __init__(self):
        super(Mul_op, self).__init__()
        self.is_unary = False
        self.is_directed = False

    def get_expr(self, sub_expr1, sub_expr2):
        return '({})*({})'.format(sub_expr1, sub_expr2)

    def transform_inputs(self, x1, x2):
        return x1 * x2


class Add_op():
    def __init__(self):
        super(Add_op, self).__init__()
        self.is_unary = False
        self.is_directed = False

    def get_expr(self, sub_expr1, sub_expr2):
        return '({})+({})'.format(sub_expr1, sub_expr2)


    def transform_inputs(self, x1, x2):
        return x1 + x2


class Div_op():
    def __init__(self, threshold=1e-10):
        super(Div_op, self).__init__()
        self.is_unary = False
        self.is_directed = True
        
        self.threshold = threshold

    def get_expr(self, sub_expr1, sub_expr2):
        return '({})/({})'.format(sub_expr1, sub_expr2)

    def transform_inputs(self, x1, x2):
        deno = torch.where(x2 < 0, torch.clamp(x2, max=-self.threshold), torch.clamp(x2, min=self.threshold))
        num = x1
        return num / deno

class Sub_op():
    def __init__(self):
        super(Sub_op, self).__init__()
        self.is_unary = False
        self.is_directed = True

    def get_expr(self, sub_expr1, sub_expr2):
        return '({})-({})'.format(sub_expr1, sub_expr2)

    def transform_inputs(self, x1, x2):
        return x1 - x2

class SemiDiv_op():
    def __init__(self, threshold=1e-10):
        super(SemiDiv_op, self).__init__()
        self.is_unary = False
        self.is_directed = False
        
        self.threshold = threshold

    def get_expr(self, sub_expr1, sub_expr2):
        return '({})/({})'.format(sub_expr1, sub_expr2)

    def transform_inputs(self, x1, x2):
        deno = torch.where(x2 < 0, torch.clamp(x2, max=-self.threshold), torch.clamp(x2, min=self.threshold))
        num = x1
        return num / deno

class SemiSub_op():
    def __init__(self):
        super(SemiSub_op, self).__init__()
        self.is_unary = False
        self.is_directed = False

    def get_expr(self, sub_expr1, sub_expr2):
        return '({})-({})'.format(sub_expr1, sub_expr2)

    def transform_inputs(self, x1, x2):
        return x1 - x2



class Sign_op():
    def __init__(self):
        super(Sign_op, self).__init__()
        self.is_unary = True
        self.is_directed = True

    def get_expr(self, sub_expr):
        return '(sign({}))'.format(sub_expr)

    def transform_inputs(self, x):
        return torch.sign(x)


class Pow2_op():
    def __init__(self):
        super(Pow2_op, self).__init__()
        self.is_unary = True
        self.is_directed = True

    def get_expr(self, sub_expr):
        return '(({})**2)'.format(sub_expr)

    def transform_inputs(self, x):
        return x ** 2


class Pow3_op():
    def __init__(self):
        super(Pow3_op, self).__init__()
        self.is_unary = True
        self.is_directed = True

    def get_expr(self, sub_expr):
        return '(({})**3)'.format(sub_expr)

    def transform_inputs(self, x):
        return x ** 3

class Pow_op():
    def __init__(self):
        super(Pow_op, self).__init__()
        self.is_unary = False
        self.is_directed = True

    def get_expr(self, sub_expr1, sub_expr2):
        return '(({})**({}))'.format(sub_expr1, sub_expr2)

    def transform_inputs(self, x1, x2):
        return x1 ** x2

class Sigmoid_op():
    def __init__(self):
        super(Sigmoid_op, self).__init__()
        self.is_unary = True
        self.is_directed = True

    def get_expr(self, sub_expr):
        return '(1/(1+exp(-({}))))'.format(sub_expr)

    def transform_inputs(self, x):
        return 1/(1 + torch.exp(-x))

class Abs_op():
    def __init__(self):
        super(Abs_op, self).__init__()
        self.is_unary = True
        self.is_directed = True

    def get_expr(self, sub_expr):
        return 'Abs({})'.format(sub_expr)

    def transform_inputs(self, x):
        return torch.abs(x)

class Cosh_op():
    def __init__(self):
        super(Cosh_op, self).__init__()
        self.is_unary = True
        self.is_directed = True

    def get_expr(self, sub_expr):
        return 'cosh({})'.format(sub_expr)

    def transform_inputs(self, x):
        return torch.cosh(x)

class Tanh_op():
    def __init__(self):
        super(Tanh_op, self).__init__()
        self.is_unary = True
        self.is_directed = True

    def get_expr(self, sub_expr):
        return 'tanh({})'.format(sub_expr)

    def transform_inputs(self, x):
        return torch.tanh(x)

class Sqrt_op():
    def __init__(self):
        super(Sqrt_op, self).__init__()
        self.is_unary = True
        self.is_directed = True

    def get_expr(self, sub_expr):
        return '({})**0.5'.format(sub_expr)

    def transform_inputs(self, x):
        return torch.sqrt(x)