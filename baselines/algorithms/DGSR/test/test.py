import torch
from nesymres.architectures.data_utils import (
    eq_remove_constants,
    eq_sympy_prefix_to_token_library,
)
from sympy import lambdify

from models.adapted_nesrymres import get_nesrymres_model

model = get_nesrymres_model()

# Create points from an equation
number_of_points = 500
n_variables = 1
target_eq = "x_1**3+x_1**2+x_1"  # Use x_1,x_2 and x_3 as independent variables
total_variables = ["x_1", "x_2", "x_3"]

# To get best results make sure that your support inside the max and mix support
max_supp = 1
min_supp = -1
X = torch.rand(number_of_points, 3) * (max_supp - min_supp) + min_supp
X[:, n_variables:] = 0
# target_eq = "sin(x_1**2)*cos(x_1)-1"
X_dict = {x: X[:, idx].cpu() for idx, x in enumerate(total_variables)}
y = lambdify(",".join(total_variables), target_eq)(**X_dict)

print("X shape: ", X.shape)
print("y shape: ", y.shape)

output, cfg = model.sample(X, y, beam_size=10)  # pyright: ignore
for e in output:
    try:
        eq = eq_remove_constants(eq_sympy_prefix_to_token_library(e))
        # variables = {x:sp.Symbol(x, real=True, nonzero=True) for x in cfg.total_variables}
        # infix = Generator.prefix_to_infix(e, coefficients=cfg.total_coefficients, variables=cfg.total_variables)
        # s = Generator.infix_to_sympy(infix,variables, cfg.rewrite_functions)
        print(eq)
        # a = test_task.library.actionize(eq)
        # p = from_tokens(a)
        # r2, acc_iid, acc_ood = compute_metrics(p)
        # print(f'stats {r2} {acc_iid} {acc_ood}')
        # p.print_stats()
    except Exception as e:  # pylint: disable=broad-exception-caught
        print(e)
