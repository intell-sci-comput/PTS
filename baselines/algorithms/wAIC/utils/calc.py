'''Output Dimension Calculator of PSRN Regressor'''
import os
import sys
current_directory = os.path.dirname(os.path.abspath(__file__))
parent_directory = os.path.dirname(current_directory)
sys.path.append(parent_directory)


from model.operators import *

n_inputs = 6 # including constant for trying e.g. ['x','sin(x)','x+x','1.2','1.5']
n_layers = 3 # PSRN layers
# operators = ['Add','Mul','SemiSub','SemiDiv','Identity','Neg','Inv','Sin','Cos','Exp','Log'] # semikoza
operators = ['Add','Mul','Sub','Div','Identity','Sin','Cos','Exp','Log']   # koza

# operators = ['Sub','Div','Sub','Div','Identity','Neg','Inv','Sin','Cos','Exp','Log'] # all_subdiv_koza

print('n_inputs',n_inputs)
print('n_layers',n_layers)
print('operators',operators)

n_u = 0
n_b_D = 0
n_b_U = 0

operators_op = []
for op_str in operators:
    op = eval(op_str+'_op')()
    operators_op.append(op)
    if op.is_unary:
        n_u += 1
    else:
        if op.is_directed:
            n_b_D += 1
        else:
            n_b_U += 1
    
x = n_inputs

for i in range(n_layers):
    x = n_u * x + n_b_U * (x * (x+1)) // 2 + n_b_D * x * x
    print('layer {} output dim {}'.format(i+1,x))

# 1.8 * 1024^5 B = 1.8 * 1024 TB 