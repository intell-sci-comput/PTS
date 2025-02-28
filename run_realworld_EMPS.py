import os
import numpy as np
import time
from utils.data import get_dynamic_data
from utils.log_ import create_dir_if_not_exist, save_pareto_frontier_to_csv
import gc
import click

@click.command()
@click.option("--gpu_index", "-g", default=0, type=int, help="gpu index used")
@click.option(
    "--n_runs", "-r", default=20, type=int, help="number of runs for each puzzle"
)
@click.option(
    "--token_generator",
    "-t",
    default="GP",
    type=str,
    help="token_generator (GP / MCTS)",
)
def main(gpu_index, n_runs, token_generator):

    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_index)

    import torch
    from model.regressor import PSRN_Regressor

    device = torch.device("cuda")
    n_seeds = n_runs
    down_sample = 200
    top_k = 30

    for seed in range(n_seeds):
        df, variables_name, target_name = get_dynamic_data("emps", "emps")

        # select the first half of the data as train set
        df = df.iloc[: len(df) // 2, :]

        gc.collect()
        np.random.seed(seed)

        Input = df[variables_name].values
        Output = df[target_name].values.reshape(-1, 1)

        p = "./log/EMPS/"

        if os.path.exists(p + "pf_{}.csv".format(seed)):
            print("exist {}, skip.".format(p + "pf_{}.csv").format(seed))
            continue

        Input = torch.from_numpy(Input).to(torch.float32)
        Output = torch.from_numpy(Output).to(torch.float32)

        regressor = PSRN_Regressor(
            variables=variables_name,
            use_const=True,
            stage_config="model/stages_config/EMPS.yaml",
            device=device,
            token_generator=token_generator,
        )

        print(Input.shape, Output.shape)
        start_time = time.time()
        flag, pareto = regressor.fit(
            Input,
            Output,
            n_down_sample=down_sample,
            use_threshold=False,
            prun_ndigit=3,
            top_k=top_k,
            add_bias=True,
        )
        end_time = time.time()
        time_cost = end_time - start_time
        print("time_cost", time_cost)

        create_dir_if_not_exist(p)
        with open(p + "time.txt", "a") as f:
            f.write(str(time_cost) + "\n")
        save_pareto_frontier_to_csv(p + "pf_{}.csv".format(seed), pareto_ls=pareto)


if __name__ == "__main__":
    main()
