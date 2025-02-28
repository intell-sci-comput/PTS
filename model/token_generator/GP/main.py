import argparse
import logging
import os
import sys
import time

import numpy as np
from pandas import read_csv

from model.config import Config
from model.pipeline import Pipeline

parser = argparse.ArgumentParser(description='GP')

parser.add_argument(
    '--task',
    default='nguyen/1',
    type=str, help="""task place""")

parser.add_argument(
    '--num_test',
    default=1,
    type=int, help='number of tests performed, default 10')

parser.add_argument(
    '--json_path',
    default="config/config.json",
    type=str, help='configuration file path')

parser.add_argument(
    "--output",
    default="output/",
    type=str, help='output path')

parser.add_argument(
    "--threshold",
    default=1e-10,
    type=float, help="threshold for evaluation")


def load_dataset(path):
    csv1 = read_csv(f"data/{path}_train.csv", header=None)
    csv2 = read_csv(f"data/{path}_test.csv", header=None)
    x, t = np.array(csv1).T[:-1], np.array(csv1).T[-1]
    x_test, t_test = np.array(csv2).T[:-1], np.array(csv2).T[-1]
    return x, t, x_test, t_test


def main(args):
    task = args.task
    num_test = args.num_test

    config = Config()
    config.json(args.json_path)
    model = Pipeline(config=config)

    all_times = []
    all_eqs = []
    num_success = 0

    data = load_dataset(task)

    for i_test in range(num_test):
        sys.stdout.flush()
        logger = logging.getLogger()
        logger.setLevel(logging.INFO)
        start_time = time.time()
        result = model.fit(*data)
        all_times.append(time.time() - start_time)
        print("\rtask:{} expr:{} RMSE:{} Test {}/{}.".format(task, result[0], result[1], i_test, num_test), end="")

        print()
        print('result')
        print(result)
        # print(' result[1]', result[1])
        # if args.threshold > result[1]:
        #     num_success += 1
        # all_eqs.append(result[0])

    output_folder = args.output
    os.makedirs(os.path.dirname(output_folder + task), exist_ok=True)
    output_file = open(output_folder + task + '.txt', 'w')
    for eq in all_eqs:
        output_file.write(eq + '\n')
    output_file.close()

    print()
    print('final result:')
    print('success rate :', "{:.0%}".format(num_success / num_test))
    print('average discovery time is', np.round(np.mean(all_times), 3), 'seconds')


if __name__ == '__main__':
    main(parser.parse_args())
