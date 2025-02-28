import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
import sympy as sp
from .model.regressor import PSRN_Regressor

def complexity(est):
    pf = est.get_pf(sort_by="mse")
    return pf[0][3]


def model(est):
    pf = est.get_pf(sort_by="mse")
    expr = pf[0][0]

    def replace_variables(expr, feature_names):
        import re

        mapping = {f"x{i}": k for i, k in enumerate(feature_names)}
        sorted_keys = sorted(mapping.keys(), key=len, reverse=True)
        pattern = "|".join(r"\b" + re.escape(k) + r"\b" for k in sorted_keys)

        def replace_func(match):
            return mapping[match.group(0)]

        new_model = re.sub(pattern, replace_func, expr)

        return new_model

    expr_sympified = sp.sympify(expr)
    expr_symplified = sp.simplify(expr_sympified)
    expr_evalf = expr_symplified.evalf(9)
    new_model = replace_variables(expr, est.feature_names)
    return new_model

est = PSRN_Regressor(variables=['x0'],
                    dr_mask_dir='./methods/SRBenchRegressor/dr_mask',
                    device='cuda',
                    token_generator='GP',
                    use_const=False,
                    use_extra_const=True,
                    token_generator_config="./methods/SRBenchRegressor/token_generator_config.yaml",
                    stage_config='./methods/SRBenchRegressor/stages.yaml',
                    set_params={
                        'use_extra_const':True,
                        'trying_const_range':[-1,1],
                        'n_down_sample':5
                        }
                    )

hyper_params = [{}]

eval_kwargs = {
    "test_params": dict(
        variables=["x"],
        device="cuda",
    )
}
