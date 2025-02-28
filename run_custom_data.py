import sys

sys.path.append(".")

import os
import click
import time
import numpy as np
import sympy as sp
import pandas as pd


from utils.data import expr_to_Y_pred


@click.command()
@click.option("--experiment_name", default="_", type=str, help="experiment_name")
@click.option("--gpu_index", "-g", default=0, type=int, help="gpu index used")
@click.option(
    "--operators",
    "-l",
    default="['Add','Mul','Sub','Div','Identity','Sin','Cos','Exp','Log']",
    help="operator library",
)
@click.option(
    "--n_down_sample",
    "-d",
    default=100,
    type=int,
    help="n sample to downsample in PSRN for speeding up",
)
@click.option(
    "--n_inputs",
    "-i",
    default=5,
    type=int,
    help="PSRN input size (n variables + n constants)",
)
@click.option("--seed", "-s", default=0, type=int, help="seed")
@click.option(
    "--topk",
    "-k",
    default=10,
    type=int,
    help="number of best expressions to take from PSRN to fit",
)
@click.option("--use_constant", "-c", default=False, type=bool, help="use const in PTS")
@click.option(
    "--probe",
    "-o",
    default=None,
    type=str,
    help="expression probe, string, PTS will stop if probe is in pf",
)
@click.option(
    "--csvpath",
    "-q",
    default="./custom_data.csv",
    type=str,
    help="path to custom csv file",
)
@click.option(
    "--token_generator",
    "-t",
    default="GP",
    type=str,
    help="token_generator (GP / MCTS)",
)
@click.option("--time_limit", default=3600, type=int, help="time limit (s)")
def main(
    experiment_name,
    gpu_index,
    operators,
    n_down_sample,
    n_inputs,
    seed,
    topk,
    use_constant,
    probe,
    csvpath,
    token_generator,
    time_limit,
):
    """

    ```
    python run_custom_data.py -g 0 -i 5 -c False --probe "(exp(x)-exp(-x))/2"
    ```

    To run the script with custom data but without an expression probe, use:
    ```
    python run_custom_data.py -g 0 -i 5 -c False
    ```

    To activate 2 constant tokens during each forward pass in PSRN, enter:
    ```
    python run_custom_data.py -g 0 -i 5 -c True -n 2 --probe "(exp(x)-exp(-x))/2"
    ```

    In case of limited VRAM (or the ground truth expression is expected to be simple), consider reducing the input size with this command:
    ```
    python run_custom_data.py -g 0 -i 2 -c False --probe "(exp(x)-exp(-x))/2"
    ```

    To customize the operator library, you can specify it like so:
    ```
    python run_custom_data.py -g 0 -i 5 -c False --probe "(exp(x)-exp(-x))/2" -l "['Add','Mul','Identity','Tanh','Abs']"
    ```
    For custom data paths, specify the CSV path as follows:
    ```
    python run_custom_data.py -g 0 -i 5 -c False --probe "(exp(x)-exp(-x))/2" --csvpath ./custom_data.csv
    ```
    """

    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_index)

    import torch
    from model.regressor import PSRN_Regressor

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    print(operators)
    operators = eval(operators)
    print(operators)

    hp = {
        "operators": operators,
        "n_down_sample": n_down_sample,
        "n_inputs": n_inputs,
        "topk": topk,
        "seed": seed,
    }

    path_log = "./log/custom_data/" + experiment_name + "/"

    if not os.path.exists(path_log):
        os.makedirs(path_log)

    cnt_success = 0
    sum_time = 0

    df = pd.read_csv(csvpath, header=None)

    variables_name = ["x"]
    target_name = ["y"]

    Input = df.values[:, :-1].reshape(len(df), -1)
    Output = df.values[:, -1].reshape(len(df), 1)

    Input = torch.from_numpy(Input).to(device).to(torch.float32)
    Output = torch.from_numpy(Output).to(device).to(torch.float32)

    print(Input.shape, Output.shape)
    print(Input.dtype, Output.dtype)

    regressor = PSRN_Regressor(
        variables=variables_name,
        dr_mask_dir="./dr_mask",
        use_const=use_constant,
        device="cuda",
        token_generator=token_generator,
        stage_config={
            "default": {
                "operators": operators,
                "time_limit": time_limit,
                "n_psrn_inputs": n_inputs,
                "n_sample_variables": 3,
            },
            "stages": [
                {},
            ],
        },
    )

    start = time.time()
    flag, pareto_ls = regressor.fit(
        Input,
        Output,
        n_down_sample=hp["n_down_sample"],
        use_threshold=False,  # Not use threshold when running benchmarks
        threshold=1e-20,
        probe=probe,  # expression probe, string, stop if probe in pf
        prun_const=True,
        prun_ndigit=2,
        top_k=topk,
    )
    end = time.time()
    time_cost = end - start

    crits = ["reward", "mse"]

    for crit in crits:
        print("Pareto Front sort by {}".format(crit))
        pareto_ls = regressor.display_expr_table(sort_by=crit)

    expr_str, reward, loss, complexity = pareto_ls[0]
    expr_sympy = sp.simplify(expr_str)

    print(expr_str)

    print("time_cost", time_cost)
    if flag:
        print("[*** Found Expr ! ***]")
        cnt_success += 1
    sum_time += time_cost

    print(expr_sympy)


if __name__ == "__main__":
    main()
