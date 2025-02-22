
import click
import os
import logging
from pathlib import Path
import pandas as pd
import torch
import torch.multiprocessing as multiprocessing
from dso.utils import log_and_print
from dso.config import load_config
from dso.task import set_task

from config import dsoconfig_factory
from config import get_config
from exp_main import top_main

@click.command()
@click.option('--exp_seed', '-s', default=0, help='seed')
@click.option('--benchmark_name', '-n', default='Constant', help='seed')
def main(exp_seed, benchmark_name):
    ls = [benchmark_name]
    cpu_num = 4
    os.environ["OMP_NUM_THREADS"] = str(cpu_num)
    os.environ["MKL_NUM_THREADS"] = str(cpu_num)
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    for benchmark_name in ls:
        
        conf = get_config()
        conf.exp.seed_runs = 1
        conf.exp.n_cores_task = 1 
        conf.exp.seed_start = exp_seed
        conf.exp.baselines = ["DGSR-PRE-TRAINED"]
        conf.exp.benchmark = benchmark_name

        torch.set_num_threads(1)

        task_name = "Main-SR" if benchmark_name != 'Constant' else 'Main-SR-Const'

        task_function_set_map = {
            "EMPS": "Koza-Sign-Const",
            "Roughpipe": "Koza-Turb-Const",
            "Main-SR-Const": "Koza-Const",
            "Main-SR": "Koza"
        }

        covar_config = task_function_set_map[task_name].lower().replace("-", "_")
        COVARS_TO_PRE_TRAINED_MODEL = {
            i: f"./models/dgsr_pre_train/{i}_covar_{covar_config}/" 
            for i in [1, 2, 3, 4, 5, 6, 8, 12]
        }

        PATH_TO_CHECK_IF_EXISTS = "./models/dgsr_pre_train/1_covar_koza/"
        Path("./logs").mkdir(parents=True, exist_ok=True)

        benchmark_df = pd.read_csv(conf.exp.benchmark_path, index_col=0, encoding="ISO-8859-1")
        df = benchmark_df[benchmark_df.index.str.contains(conf.exp.benchmark)]
        file_name = os.path.basename(os.path.realpath(__file__)).split(".py")[0]
        path_run_name = "all_{}-{}_01".format(file_name, conf.exp.benchmark)

        def create_our_logger(path_run_name):
            logger = multiprocessing.get_logger()
            formatter = logging.Formatter("%(processName)s| %(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s")
            stream_handler = logging.StreamHandler()
            file_handler = logging.FileHandler("./logs/{}_log.txt".format(path_run_name))
            stream_handler.setFormatter(formatter)
            file_handler.setFormatter(formatter)
            logger.addHandler(stream_handler)
            logger.addHandler(file_handler)
            logger.setLevel(logging.INFO)
            logger.info("STARTING NEW RUN ==========")
            logger.info(f"SEE LOG AT : ./logs/{path_run_name}_log.txt")
            return logger

        logger = create_our_logger(path_run_name)
        logger.info(f"See log at : ./logs/{path_run_name}_log.txt")
        print(df)
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

        def main(dataset, n_cores_task=conf.exp.n_cores_task):
            if not os.path.exists(PATH_TO_CHECK_IF_EXISTS):
                print("Downloading pre-trained models and results for the first and only time... (this may take a few minutes)")
                from download_pre_trained_models import download_pre_trained_models_and_results

                download_pre_trained_models_and_results()
            task_inputs = []
            for seed in range(conf.exp.seed_start, conf.exp.seed_start + conf.exp.seed_runs):
                for baseline in conf.exp.baselines:
                    task_inputs.append((seed, dataset, pre_trained_model, baseline))

            if n_cores_task is None:
                n_cores_task = multiprocessing.cpu_count()
            if n_cores_task >= 2:
                pool_outer = multiprocessing.Pool(n_cores_task)
                for i, result in enumerate(pool_outer.imap(perform_run, task_inputs)):
                    log_and_print(
                        "INFO: Completed run {} of {} | LATEST TEST_RESULT {}".format(
                            i + 1, len(task_inputs),  result
                        )
                    )
            else:
                for i, task_input in enumerate(task_inputs):
                    result = perform_run(task_input)
                    log_and_print(
                        "INFO: Completed run {} of {} | LATEST TEST_RESULT {}".format(
                            i + 1, len(task_inputs), result
                        )
                    )

        if __name__ == "__main__":
            torch.multiprocessing.set_start_method("spawn")

            dsoconfig = dsoconfig_factory()
            log_and_print(df.to_string())
            for dataset, row in df.iterrows():
                covars = row["variables"]
                try:
                    pre_trained_model = COVARS_TO_PRE_TRAINED_MODEL[covars]
                except KeyError:
                    # pylint: disable-next=raise-missing-from
                    raise ValueError(
                        f"No pre-trained model in folder './models/pre_train/' for covars={covars}. "
                        "Please download the pre-trained models. See README.md for more details. "
                        "Or alternatively, run 'download_pre_trained_models.py' to download the pre-trained models."
                    )
                    # pre_trained_model = ""
                dsoconfig["task"]["dataset"] = dataset
                config = load_config(dsoconfig)
                set_task(config["task"])
                try:
                    main(dataset)
                except FileNotFoundError as e:
                    # pylint: disable-next=raise-missing-from
                    raise FileNotFoundError(
                        f"No pre-trained model of {e.filename} in folder './models/pre_train/' for covars={covars}. "
                        "Please download the pre-trained models. See README.md for more details. Or alternatively, "
                        "run 'download_pre_trained_models.py' to download the pre-trained models."
                    )
            logger.info("Fin.")

if __name__ == '__main__':
    main()