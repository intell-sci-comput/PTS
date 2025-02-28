import sys

sys.path.append(".")

import os
import click
import time
import numpy as np
import sympy as sp
import pandas as pd

import gc
import traceback


def if_is_exist(path, name):
    for root, dirs, files in os.walk(path):
        for file in files:
            if name in file:
                return True
    return False


testlist = [
    "Nguyen-1c",
    "Nguyen-2c",
    "Nguyen-5c",
    "Nguyen-8c",
    "Nguyen-9c",
    "Nguyen-10c",
]


@click.command()
@click.option("--experiment_name", "-x", default="_", type=str, help="experiment_name")
@click.option(
    "--n_runs", "-n", default=20, type=int, help="The number of runs per benchmark"
)
@click.option("--gpu_index", "-g", default=0, type=int, help="gpu index used")
@click.option(
    "--n_down_sample",
    "-d",
    default=50,
    type=int,
    help="n sample to downsample in PSRN for speeding up",
)
@click.option("--seed", "-s", default=0, type=int, help="seed")
@click.option(
    "--benchmark_file",
    "-b",
    default="benchmark.csv",
    type=str,
    help="benchmark csv name",
)
@click.option(
    "--const_range", "-c", default=False, type=str, help="const sampling range"
)
@click.option(
    "--token_generator",
    "-t",
    default="GP",
    type=str,
    help="token_generator (GP / MCTS)",
)
def main(
    experiment_name,
    n_runs,
    gpu_index,
    n_down_sample,
    seed,
    benchmark_file,
    const_range,
    token_generator,
):

    const_range = eval(const_range)
    for i in range(len(const_range)):
        const_range[i] = eval(const_range[i])
    print(const_range)

    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_index)

    import torch
    from model.regressor import PSRN_Regressor
    from utils.data import get_benchmark_data, expr_to_Y_pred

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    random_seed = seed

    hp = {
        "const_range": const_range,
        "n_down_sample": n_down_sample,
        "seed": seed,
    }

    for key, value in hp.items():
        experiment_name += "_{}".format(value)

    ########### load data #############

    path_log = "./log/" + experiment_name + "/"
    if not os.path.exists(path_log):
        os.makedirs(path_log)

    df = pd.read_csv("./benchmark/{}".format(benchmark_file))
    df_save_all = pd.DataFrame(
        columns=["name", "recovery_rate", "avg_time_cost", "n_runs"]
    )
    err_txt_path = path_log + "error_file.txt"

    for benchmark_name in df["name"]:

        if benchmark_name not in testlist:
            continue

        if if_is_exist(path_log[:-1], benchmark_name):
            continue

        # try:
        print("Runing benchmark: {}".format(benchmark_name))

        cnt_success = 0
        sum_time = 0

        print("n_runs: {}".format(n_runs))

        df_save = pd.DataFrame(
            columns=[
                "name",
                "success",
                "time_cost",
                "expr_str",
                "expr_sympy",
                "R2",
                "MSE",
                "reward",
                "complexity",
            ]
        )

        for i in range(n_runs):

            gc.collect()

            np.random.seed(random_seed + i)

            print("Runing {}-th time".format(i + 1))

            Input, Output, use_constant, expression, variables_name = (
                get_benchmark_data(benchmark_file, benchmark_name)
            )

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
            )

            regressor.trying_const_range = const_range
            print("regressor.trying_const_range")
            print(regressor.trying_const_range)

            start = time.time()
            flag, pareto_ls = regressor.fit(
                Input, Output, probe=expression, prun_const=True, prun_ndigit=2
            )
            end = time.time()
            time_cost = end - start

            crits = ["mse", "reward"]

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

            df_save = df_save.append(
                {
                    "name": benchmark_name,
                    "success": flag,
                    "time_cost": time_cost,
                    "expr_str": expr_str,
                    "expr_sympy": expr_sympy,
                    "R2": None,
                    "MSE": loss,
                    "reward": reward,
                    "complexity": complexity,
                },
                ignore_index=True,
            )

            df_hof = pd.DataFrame(
                pareto_ls, columns=["expr_str", "reward", "MSE", "complexity"]
            )
            df_hof = df_hof.head(10)
            t = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())

            if not os.path.exists(path_log + "/{}".format(benchmark_name)):
                os.makedirs(path_log + "/{}".format(benchmark_name))
            df_hof.to_csv(
                path_log + "{}/hof_{}_{}.csv".format(benchmark_name, i, t), index=False
            )

        # except Exception as e:
        #     traceback_info = traceback.format_exc()

        #     with open(err_txt_path, "a") as f:
        #         f.write(str(traceback_info))

        t = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
        df_save.to_csv(
            path_log + "benchmark_{}_{}.csv".format(benchmark_name, t), index=False
        )

        avg_time = sum_time / n_runs
        avg_success_rate = cnt_success / n_runs
        df_save_all = df_save_all.append(
            {
                "name": benchmark_name,
                "recovery_rate": avg_success_rate,
                "avg_time_cost": avg_time,
                "n_runs": n_runs,
            },
            ignore_index=True,
        )

        print(df_save_all)

    t = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
    df_save_all.to_csv(path_log + "benchmark_all_{}.csv".format(t))


if __name__ == "__main__":
    main()
