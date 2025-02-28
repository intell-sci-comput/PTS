# check
import os
import gc

import numpy as np
import time
from dysts.flows import *
from pysindy import SmoothedFiniteDifference
from utils.data import add_noise
from utils.log_ import create_dir_if_not_exist, save_pareto_frontier_to_csv

import click


@click.command()
@click.option("--gpu_index", "-g", default=0, type=int, help="gpu index used")
@click.option(
    "--n_runs", "-r", default=50, type=int, help="number of runs for each puzzle"
)
@click.option(
    "--token_generator",
    "-t",
    default="GP",
    type=str,
    help="token_generator (GP / MCTS)",
)
@click.option("--run_1min", "-m", default=False, type=bool, help="run_1min")
def main(gpu_index, n_runs, token_generator, run_1min):

    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_index)
    import torch
    from model.regressor import PSRN_Regressor

    device = torch.device("cuda")

    class_name_ls = open("./data/dystsnames/dysts_16_flows.txt").readlines()
    class_name_ls = [line.strip() for line in class_name_ls]

    print(class_name_ls)

    n_seeds = n_runs
    sample_size = 1000
    pts_per_period = 100
    noise_level = 0.01
    down_sample = 200

    top_k = 30
    postprocess = False

    for seed in range(n_seeds):

        for class_name in class_name_ls:

            # Convert to autonomous system
            dysts_model = eval(class_name)
            dysts_model = dysts_model()
            if class_name == "ForcedBrusselator":
                dysts_model.f = 0
            elif class_name == "Duffing":
                dysts_model.gamma = 0
            elif class_name == "ForcedVanDerPol":
                dysts_model.a = 0
            elif class_name == "ForcedFitzHughNagumo":
                dysts_model.f = 0

            # clear PSRN's VRAM memory
            gc.collect()
            np.random.seed(seed)  # setting random seed
            try:
                print("=== START SEARCHING FOR GT MODEL: ===")
                print(dysts_model)
                # Gen traj data using `dysts` library
                t, data = dysts_model.make_trajectory(
                    sample_size,
                    return_times=True,
                    pts_per_period=pts_per_period,
                    resample=True,
                    noise=0,
                    postprocess=postprocess,
                )

                # introduce gaussian noise
                for k in range(data.shape[1]):
                    data[:, k : k + 1] = add_noise(
                        data[:, k : k + 1], noise_level, seed
                    )

                # Make derivation for each variable
                t = t.flatten()
                dim = data.shape[1]
                print("dim", dim)
                if dim == 3:
                    variable_names = ["x", "y", "z"]
                    dotsvarnames = ["xdot", "ydot", "zdot"]
                    deriv_idxs = [0, 1, 2]
                elif dim == 4:
                    variable_names = ["x", "y", "z", "w"]
                    dotsvarnames = ["xdot", "ydot", "zdot", "wdot"]
                    deriv_idxs = [0, 1, 2, 3]
                else:
                    continue

                sfd = SmoothedFiniteDifference(smoother_kws={"window_length": 5})
                data_deriv = np.zeros((data.shape[0], len(deriv_idxs)))
                for i, idx in enumerate(deriv_idxs):
                    print(data[:, idx : idx + 1].shape, t.shape)
                    deriv_data_i = sfd._differentiate(data[:, idx : idx + 1], t)
                    data_deriv[:, i : i + 1] = deriv_data_i

                for idxdot, vardotname in enumerate(dotsvarnames):

                    p = "./log/chaotic_1min={}/{}/{}/".format(
                        run_1min, class_name, vardotname
                    )

                    if os.path.exists(p + "pf_{}.csv".format(seed)):
                        print("exist {}, skip.".format(p + "pf_{}.csv").format(seed))
                        continue

                    gc.collect()
                    torch.cuda.empty_cache()

                    Input = torch.from_numpy(data).to(torch.float32).to(device)
                    Output = (
                        torch.from_numpy(data_deriv[:, idxdot : idxdot + 1])
                        .to(torch.float32)
                        .to(device)
                    )

                    if not run_1min:
                        yml_p = "model/stages_config/chaotic.yaml"
                    else:
                        yml_p = "model/stages_config/chaotic_1min.yaml"

                    regressor = PSRN_Regressor(
                        variables=variable_names,
                        use_const=True,
                        stage_config=yml_p,
                        device=device,
                    )
                    print(Input.shape, Output.shape)

                    start_time = time.time()
                    flag, pareto = regressor.fit(
                        Input,
                        Output,
                        n_down_sample=down_sample,
                        use_threshold=False,
                        prun_ndigit=2,
                        top_k=top_k,
                    )
                    end_time = time.time()
                    time_cost = end_time - start_time
                    print("time_cost", time_cost)

                    create_dir_if_not_exist(p)
                    with open(p + "time.txt", "a") as f:
                        f.write(str(time_cost) + "\n")
                    save_pareto_frontier_to_csv(
                        p + "pf_{}.csv".format(seed), pareto_ls=pareto
                    )

            except np.linalg.LinAlgError:
                print("np.linalg.LinAlgError (Intel bug)")
                continue


if __name__ == "__main__":
    main()
