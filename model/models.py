
import torch
import torch.nn as nn

from model.functions import (Identity,
                             Sin,
                             Cos,
                             Exp,
                             Log,
                             Neg,
                             Inv,
                             Add,
                             Mul,
                             Div,
                             Sub,
                             SemiDiv,
                             SemiSub)

from model.functions import (Sign,
                             Pow2,
                             Pow3,
                             Pow,
                             Sigmoid,
                             Abs,
                             Cosh,
                             Tanh,
                             Sqrt)
        
# Duplicate Removal Layer
class DRLayer(nn.Module):
    def __init__(self,
                 in_dim,
                 dr_mask,
                 device=None):
        super(DRLayer, self).__init__()
        
        self.in_dim = in_dim
        self.out_dim = round(torch.sum(dr_mask).item())
        arange_tensor = torch.arange(len(dr_mask),device=device)
        self.dr_indices = arange_tensor[dr_mask] # (n,)
        self.dr_mask = dr_mask                                # (n,)
        
        self.dr_indices = self.dr_indices.to(device)
        self.dr_mask = self.dr_mask.to(device)
    
    def forward(self, x):
        # shape x: (batch_size, in_dim)
        return x[:, self.dr_mask]

    def get_op_and_offset(self, index):
        return self.dr_indices[index].item()
        
        

class SymbolLayer(nn.Module):
    def __init__(self,
                 in_dim,
                 operators=['Add', 'Mul', 'Identity',
                            'Sin', 'Exp', 'Neg', 'Inv'],
                 device=None):
        super(SymbolLayer, self).__init__()
        

        self.device = device

        self.in_dim = in_dim
        self.n_triu = (in_dim * (in_dim + 1) // 2)
        self.in_dim_square = in_dim * in_dim
        self.operators = operators

        self.list = nn.ModuleList()
        self.n_binary_U = 0 # undirected * +
        self.n_binary_D = 0 # directed   / -
        self.n_unary = 0
        
        for op in operators:
            func = eval(op)(in_dim, device)
            if not func.is_unary:
                if func.is_directed:
                    self.n_binary_D += 1
                else:
                    self.n_binary_U += 1
            else:
                self.n_unary += 1
            # self.list.append(func)
        # first place Add and Mul (triangled-shaped ops)
        for op in operators:
            func = eval(op)(in_dim, device)
            if not func.is_unary and not func.is_directed:
                self.list.append(func)
        
        # then place Sub and Div (squared-shape ops)
        for op in operators:
            func = eval(op)(in_dim, device)
            if not func.is_unary and func.is_directed:
                self.list.append(func)
                
        # finally unary ops
        for op in operators:
            func = eval(op)(in_dim, device)
            if func.is_unary:
                self.list.append(func)     

        self.out_dim = self.n_unary * self.in_dim + self.n_binary_U * self.n_triu + self.n_binary_D * self.in_dim_square

        self.out_dim_cum_ls = None
        self.init_offset(device)

    def forward(self, x):
        # shape x: (batch_size, in_dim)
        h = []
        for module in self.list:
            h.append(module(x))
        h = torch.cat(h, dim=1)  # shape: (batch_size, out_dim)
        return h

    def init_offset(self, device):
        self.offset_tensor = self.get_offset_tensor(device)

    def get_offset_tensor(self, device):
        offset_tensor = torch.zeros(
            (self.out_dim, 2), dtype=torch.int, device=device)
        arange_tensor = torch.arange(self.in_dim, dtype=torch.int, device=device)
        
        binary_U_tensor = torch.zeros(
            (self.n_triu, 2), dtype=torch.int, device=device)
        binary_D_tensor = torch.zeros(
            (self.in_dim_square, 2), dtype=torch.int, device=device)
        unary_tensor = torch.zeros((self.in_dim, 2), dtype=torch.int, device=device)

        unary_tensor[:, 0] = arange_tensor
        unary_tensor[:, 1] = self.in_dim

        start = 0
        for i in range(self.in_dim):
            len_ = self.in_dim - i
            binary_U_tensor[start:start + len_, 0] = i
            binary_U_tensor[start:start + len_, 1] = arange_tensor[i:]
            start += len_

        start = 0
        for i in range(self.in_dim):
            len_ = self.in_dim
            binary_D_tensor[start:start + len_, 0] = i
            binary_D_tensor[start:start + len_, 1] = arange_tensor[0:]
            start += len_

        start = 0
        for func in self.list:
            if not func.is_unary:
                if func.is_directed:
                    t = binary_D_tensor
                else:
                    t = binary_U_tensor
            else:
                t = unary_tensor
            len_ = t.shape[0]

            offset_tensor[start:start + len_:] = t
            start += len_

        return offset_tensor

    def get_out_dim_cum_ls(self):
        if self.out_dim_cum_ls != None:
            return self.out_dim_cum_ls

        out_dim_ls = []
        for func in self.list:
            if not func.is_unary:
                if func.is_directed:
                    out_dim_ls.append(self.in_dim_square)
                else:
                    out_dim_ls.append(self.n_triu)
            else:
                out_dim_ls.append(self.in_dim)
        self.out_dim_cum_ls = [sum(out_dim_ls[:i+1])
                               for i in range(len(out_dim_ls))]
        return self.out_dim_cum_ls

    def get_op_and_offset(self, index):
        out_dim_cum_ls = self.get_out_dim_cum_ls()        
        for i, func in enumerate(self.list):
            if index < out_dim_cum_ls[i]:
                break
        offset = self.offset_tensor[index].tolist() # index 65 is out of bounds for dimension 0 with size 65
        return func.operator, offset


class PSRN(nn.Module):
    def __init__(self,
                 n_variables=1,
                 operators=['Add', 'Mul', 'Identity',
                            'Sin', 'Exp', 'Neg', 'Inv'],
                 n_symbol_layers=3,
                 dr_mask=None,
                 device='cuda'):
        super(PSRN, self).__init__()

        if isinstance(device, str):
            if device == 'cuda':
                self.device = torch.device('cuda')
            elif device == 'cpu':
                self.device = torch.device('cpu')
            else:
                raise ValueError(
                    'device must be cuda or cpu, got {}'.format(device))
        self.device = device
        self.n_variables = n_variables
        self.operators = operators
        self.n_symbol_layers = n_symbol_layers
        
        self.list = nn.ModuleList()
        
        if dr_mask is None:
            self.use_dr_mask = False
        else:
            self.use_dr_mask = True
        
        if self.use_dr_mask:
            assert type(dr_mask) == torch.Tensor, 'dr_mask must be a tensor'
            assert dr_mask.dim() == 1, 'dr_mask should be 1-dim, got {}'.format(dr_mask.dim())
            dr_mask = dr_mask.to(self.device)


        for i in range(n_symbol_layers):
            if self.use_dr_mask and i == n_symbol_layers - 1:
                self.list.append(DRLayer(
                    self.list[-1].out_dim, dr_mask=dr_mask, device=self.device))
            
            if i == 0:
                self.list.append(SymbolLayer(
                    n_variables, operators, device=self.device))
            else:
                self.list.append(SymbolLayer(
                    self.list[-1].out_dim, operators, device=self.device))

        self.current_expr_ls = []

        self.out_dim = self.list[-1].out_dim

    def __repr__(self):
        return super().__repr__() + '\n' + 'n_inputs: {}, operators: {}, n_layers: {}'.format(
            self.n_variables,
            self.operators,
            self.n_symbol_layers) + '\n dim:' + '\n'.join(str(layer.out_dim) for layer in self.list)
    
    def forward(self, x):
        # shape x: (batch_size, n_variables)
        h = x
        for i, layer in enumerate(self.list):
            h = layer(h)
        return h  # shape: (batch_size, out_dim)

    def get_expr(self, index):
        return self._get_expr(index, -1)

    def _get_expr(self, index, layer_idx):

        if len(self.list) + layer_idx < 0:
            return self.current_expr_ls[index]

        layer = self.list[layer_idx]
        
        if layer._get_name() == 'DRLayer':
            new_index = layer.get_op_and_offset(index)
            return self._get_expr(new_index, layer_idx-1)
        
        else:
            # SymbolLayer
     
            func_op, offset = layer.get_op_and_offset(index)

            if func_op.is_unary:  
                return func_op.get_expr(
                    self._get_expr(offset[0], layer_idx-1)
                    )
            else:
                return func_op.get_expr(
                    self._get_expr(offset[0], layer_idx-1),
                    self._get_expr(offset[1], layer_idx-1),
                    )