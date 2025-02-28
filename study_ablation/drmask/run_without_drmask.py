import sys

sys.path.append(".")

import os
import click
import time
import numpy as np


@click.command()
@click.option(
    "--use_drmask", "-u", default=False, type=bool, help="whether to use drmask"
)
@click.option("--gpu_idx", "-g", default=0, type=int, help="gpu_idx")
@click.option("--n_inputs", "-i", default=5, type=int, help="number of input of psrn")
@click.option("--library", "-l", default="koza", type=str, help="operator library")
def main(use_drmask, gpu_idx, n_inputs, library):
    """
    >>> python study_ablation/drmask/run_without_drmask.py --use_drmask True -g 0

    >>> python study_ablation/drmask/run_without_drmask.py --use_drmask False -g 0

    ########### VRAM usages: ##########
    # input  operators     use_drmaks     VRAM
    # 4      koza          False          27.22 GB
    # 4      koza          True           10.65 GB

    # 5      koza          False          [OOM]
    # 5      koza          True           46.22 GB

    # 6      semi_koza     False          75.07 GB
    # 6      semi_koza     True           37.79 GB

    # 10     basic         False          43.19 GB
    # 10     basic         True           33.84 GB

    """

    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_idx)  # set

    import torch
    from model.regressor import PSRN_Regressor

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    Input = np.random.rand(100, 1)
    Output = np.random.rand(100, 1)

    Input = torch.from_numpy(Input).to(device).to(torch.float32)
    Output = torch.from_numpy(Output).to(device).to(torch.float32)

    print(Input.shape, Output.shape)
    print(Input.dtype, Output.dtype)

    hp = {
        "use_dr_mask": use_drmask,
        "seed": 0,
    }

    lib2operators = {
        "koza": ["Add", "Mul", "Sub", "Div", "Identity", "Sin", "Cos", "Exp", "Log"],
        "semikoza": [
            "Add",
            "Mul",
            "SemiSub",
            "SemiDiv",
            "Identity",
            "Neg",
            "Inv",
            "Sin",
            "Cos",
            "Exp",
            "Log",
        ],
        "basic": ["Add", "Mul", "Identity", "Neg", "Inv", "Sin", "Cos", "Exp", "Log"],
    }

    operators = lib2operators[library]

    regressor = PSRN_Regressor(
        variables=["x"],
        use_dr_mask=hp["use_dr_mask"],
        use_const=False,
        stage_config={
            "default": {
                "operators": operators,
                "time_limit": 60,
                "n_psrn_inputs": n_inputs,
                "n_sample_variables": 3,
            },
            "stages": [
                {},
            ],
        },
        device=device,
    )
    print(Input.shape, Output.shape)

    start = time.time()
    flag, pareto = regressor.fit(Input, Output)

    end = time.time()
    time_cost = end - start


if __name__ == "__main__":
    main()
