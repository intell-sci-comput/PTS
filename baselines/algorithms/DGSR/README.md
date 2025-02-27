# Deep Generative Symbolic Regression (Code)

This repository is the official implementation of the paper [Deep Generative Symbolic Regression](https://openreview.net/forum?id=o7koEEMA1bR).


1. Run/Follow steps in [install.sh](install.sh)
2. Replicate experimental results by running and configuring [run_recovery_multi.py](run_recovery_multi.py)
3. Process the output log file using [process_logs.py](process_results/process_logs.py) by updating the `LOG_PATH` variable to point to the recently generated log file



#### Large files:
To automatically download and extract the pre-trained models into the right folders run the file [download_pre_trained_models.py](download_pre_trained_models.py) from the root directory of this repository. This will download the pre-trained models and place them into the correct folders, including the results used for the paper in [`process_results/results/`](process_results/results/).

Alternatively, this can be manually downloaded and unzipped from the following link https://drive.google.com/file/d/1tZcyDKT2GeP35fTnnddiiUC3ow6OVcIF



#### Configuring experiments:
The simplest way to configure the experiments is to modify the following parameters at the top of [run_recovery_multi.py](run_recovery_multi.py)
```python
conf.exp.seed_runs = 40
conf.exp.n_cores_task = 1  # 7 if GPU memory is at least 24GB, else tune to be smaller
conf.exp.seed_start = 0
conf.exp.baselines = ["DGSR-PRE-TRAINED", "NGGP", "NESYMRES", "GP"]
# User must specify the benchmark to run:
conf.exp.benchmark = "fn_d_2"  # Possible values ["fn_d_2", "fn_d_5", "l_cd_12", ""fn_d_all"]
```
The final parameter `conf.exp.benchmark` is the most important, as it corresponds to which benchmark experiment is to be run. Select one of the possible values specified in the comment.



#### Re-training the models for a new dataset or equation class of new input variables of token set library:

First configure and then run the file [run_pretrain.py](run_pretrain.py). This file is used to pre-train the DGSR model for a specific dataset that uses a specific number of input variables (covariates) as defined by the test dataset equation selected. This will train the model essentially forever, and continually saves the model during training. After a period of 4-5 hours, the model performance loss will plateau. The user will need to then manually stop the training process, and copy the saved model (`controller.pt`) and model configuration file (`config.json`) from the log directory (`./log/{$RUN}`, where `$RUN` is a folder name that includes the trained dataset and date as a string) to the pre-trained models directory (`./models/dgsr_pre-train/`), and then update the path to load when testing in `run_recovery_multi.py` in the dict `COVARS_TO_PRE_TRAINED_MODEL`.

<!-- Sam ToDo before release: Put back in Feynman_d_Feynman_4 eq, clean up all code, and benchmarks, rename DSO to NGGP, and OURS to DGSR -->
