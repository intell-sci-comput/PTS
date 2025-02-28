import sys

sys.path.append(".")

import click
import os
import numpy as np

command_ls = []


@click.command()
@click.option("--gpu_idx", "-g", default=0, type=int, help="gpu index")
@click.option("--n_runs", "-n", default=20, type=int, help="gpu index")
@click.option(
    "--token_generator",
    "-t",
    default="GP",
    type=str,
    help="token_generator (GP / MCTS)",
)
def main(gpu_idx, n_runs, token_generator):

    command_head = (
        """python study_ablation/constants/const.py -x const_range"""
        + token_generator
        + """ -n {} -g {} -c""".format(n_runs, gpu_idx)
    )
    # np random seed

    np.random.seed(0)

    for n_try in [1, 3, 10]:
        bound = [-n_try, n_try]
        for i in range(len(bound)):
            bound[i] = str(round(bound[i], 1))
        bound = str(bound)
        bound.replace(" ", "")

        command = command_head + ' "' + bound + '"' + " -t " + token_generator
        print(command)
        command_ls.append(command)

    for command in command_ls:
        print("Executing:")
        print(command)
        os.system(command)


if __name__ == "__main__":
    main()
