import os
os.environ['CUDA_VISIBLE_DEVICES'] = '2'
import gc
import time
import click
import torch
import numpy as np
import pandas as pd
from pathlib import Path
import torch.multiprocessing as multiprocessing
import logging
from dysts.flows import *
from pysindy import SmoothedFiniteDifference
from utils.data import add_noise
from dso.utils import log_and_print
from config import get_config
from exp_main import top_main

task_name = "Chaotic"

@click.command()
@click.option('--n_runs', '-r', default=50, type=int, help='Number of runs for each puzzle')
@click.option('--exp_seed', '-s', default=0, help='Seed')
def main(n_runs, exp_seed):

    class_name_ls = open('../../../data/dystsnames/dysts_16_flows.txt').readlines()
    class_name_ls = [line.strip() for line in class_name_ls]

    print(class_name_ls)

    # DYSTS settings
    n_seeds = n_runs
    sample_size = 1000
    pts_per_period = 100
    noise_level = 0.01

    # Symbolic regression settings
    exp_seed = exp_seed

    conf = get_config()
    conf.exp.seed_runs = 1
    conf.exp.n_cores_task = 1
    conf.exp.seed_start = exp_seed
    conf.exp.baselines = ["DGSR-PRE-TRAINED"]
    task_function_set_map = {
        "EMPS": "Koza-Sign-Const",
        "Roughpipe": "Koza-Turb-Const",
        "Chaotic": "Koza-Chaotic-Const",
        "Main-SR-Const": "Koza-Const",
        "Main-SR": "Koza"
    }
    covar_config = task_function_set_map[task_name].lower().replace("-", "_")
    COVARS_TO_PRE_TRAINED_MODEL = {
        i: f"./models/dgsr_pre_train/{i}_covar_{covar_config}/" 
        for i in [1, 2, 3, 4, 5, 6, 8, 12]
    }


    Path("./logs").mkdir(parents=True, exist_ok=True)
    benchmark_df = pd.read_csv(conf.exp.benchmark_path, index_col=0, encoding="ISO-8859-1")
    df = benchmark_df[benchmark_df.index.str.contains(conf.exp.benchmark)]
    def create_our_logger(path_run_name):
        logger = multiprocessing.get_logger()
        formatter = logging.Formatter("%(processName)s| %(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s")
        stream_handler = logging.StreamHandler()
        file_handler = logging.FileHandler(f"./logs/{path_run_name}_log.txt")
        stream_handler.setFormatter(formatter)
        file_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)
        logger.addHandler(file_handler)
        logger.setLevel(logging.INFO)
        logger.info("STARTING NEW RUN ==========")
        logger.info(f"SEE LOG AT : ./logs/{path_run_name}_log.txt")
        return logger

    data_samples_to_use = int(float(df["train_spec"][0].split(",")[-1].split("]")[0]) * conf.exp.dataset_size_multiplier)
    def perform_run(tuple_in):
        seed, dataset, pre_trained_model, baseline = tuple_in
        logger.info(
            f"[BASELINE_TESTING NOW] dataset={dataset} \t| baseline={baseline} \t| seed={seed} \t| data_samples={data_samples_to_use} \t| noise={conf.exp.noise}"
        )
        result = top_main(
                test_dataset=dataset,
                seed=seed,
                training_equations=200000,
                training_epochs=100,
                batch_outer_datasets=24,
                batch_inner_equations=100,
                pre_train=True,
                skip_pre_training=True,
                load_pre_trained_path=pre_trained_model,
                priority_queue_training=conf.exp.priority_queue_training,
                gp_meld=conf.gp_meld.run_gp_meld,
                model="TransformerTreeEncoderController",
                train_path="",
                test=conf.exp.run_pool_programs_test,
                risk_seeking_pg_train=True,
                save_true_log_likelihood=conf.exp.save_true_log_likelihood,
                p_crossover=conf.gp_meld.p_crossover,
                p_mutate=conf.gp_meld.p_mutate,
                tournament_size=conf.gp_meld.tournament_size,
                generations=conf.gp_meld.generations,
                function_set=conf.exp.function_set,
                learning_rate=conf.exp.learning_rate,
                test_sample_multiplier=conf.exp.test_sample_multiplier,
                n_samples=conf.exp.n_samples,
                dataset_size_multiplier=conf.exp.dataset_size_multiplier,
                noise=conf.exp.noise,
                function_set_name=task_function_set_map[task_name]
            )
        result["baseline"] = baseline
        result["run_seed"] = seed
        result["dataset"] = dataset
        log_and_print(f"[TEST RESULT] {result}")
        return result

    for seed in range(n_seeds):
        for class_name in class_name_ls:
            # Convert to autonomous system
            dysts_model = eval(class_name)()
            if class_name == 'ForcedBrusselator':
                dysts_model.f = 0
            elif class_name == 'Duffing':
                dysts_model.gamma = 0
            elif class_name == 'ForcedVanDerPol':
                dysts_model.a = 0
            elif class_name == 'ForcedFitzHughNagumo':
                dysts_model.f = 0
            
            gc.collect()
            np.random.seed(seed)
            
            print('=== START SEARCHING FOR GT MODEL: ===')
            print(dysts_model) 
            
            # Generate trajectory data
            t, data = dysts_model.make_trajectory(sample_size,
                                                  return_times=True,
                                                  pts_per_period=pts_per_period,
                                                  resample=True,
                                                  noise=0,
                                                  postprocess=False)
           # Add noise
            for k in range(data.shape[1]):
                data[:,k:k+1] = add_noise(data[:,k:k+1], noise_level, seed)

            # Prepare data for symbolic regression
            dim = data.shape[1]
            if dim == 3:
                vars = ['x','y','z']
                dotsvarnames = ['xdot','ydot','zdot']
                deriv_idxs = [0,1,2]
            elif dim == 4:
                vars = ['x','y','z','w']
                dotsvarnames = ['xdot','ydot','zdot','wdot']
                deriv_idxs = [0,1,2,3]
            else:
                continue
            
            sfd = SmoothedFiniteDifference(smoother_kws={'window_length': 5})
            data_deriv = np.zeros((data.shape[0],len(deriv_idxs)))
            for i,idx in enumerate(deriv_idxs):
                deriv_data_i = sfd._differentiate(data[:,idx:idx+1], t)
                data_deriv[:,i:i+1] = deriv_data_i

            # Create separate CSV files for each output variable
            for i, dotvarname in enumerate(dotsvarnames):
                p = './data/chaotic/'
                if not os.path.exists(p):
                    os.makedirs(p)
                csv_path = p+f'dysts_{class_name}_{seed}_{dotvarname}.csv'
                
                # Check if this configuration has already been run
                existing_logs = [d for d in os.listdir("./logs") if d.startswith(f"dysts10min_{class_name}_{seed}_{dotvarname}_")]
                if existing_logs:
                    print(f"Skipping {class_name} seed {seed} {dotvarname} as it has already been run.")
                    continue
                
                # Prepare input columns (4 columns)
                input_data = data[:, :4] if dim == 4 else np.hstack([data, data[:, -1:]])  # Pad with last column if dim == 3
                
                # Prepare output column
                output_data = data_deriv[:, i:i+1]
                
                # Combine input and output
                combined_data = np.hstack([input_data, output_data])
                
                # Create column names
                column_names = vars[:4] if dim == 4 else vars + [vars[-1]]  # Pad with last variable if dim == 3
                column_names.append(dotvarname)
                
                # Save to CSV
                df = pd.DataFrame(combined_data, columns=column_names)
                df.to_csv(csv_path, index=False, header=None)

                # Run symbolic regression
                logger = create_our_logger(f"dysts10min_{class_name}_{seed}_{dotvarname}")
                
                pre_trained_model = COVARS_TO_PRE_TRAINED_MODEL[1]  # Always use 4 covars model
                baseline = "DGSR-PRE-TRAINED"
                
                task_inputs = (seed, csv_path, pre_trained_model, baseline)
                result = perform_run(task_inputs)
                
                try:
                    log_and_print(
                        f"INFO: Completed run for {class_name} (seed {seed}, {dotvarname}) in {result['t']:.0f} s | LATEST TEST_RESULT {result}"
                    )
                except Exception as e:
                    log_and_print(f"Error in logging result: {e}")

if __name__ == '__main__':
    # torch.multiprocessing.set_start_method("spawn")
    main()