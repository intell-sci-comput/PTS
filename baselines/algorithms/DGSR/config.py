import collections

from omegaconf import OmegaConf


def dsoconfig_factory(function_set="Koza"):
    print('='*40)
    print('function_set:', function_set)
    print('='*40)
    
    function_set_dict = {
        "Koza":              ["add", "sub", "mul", "div", "sin", "cos", "exp", "log"],
        "Koza-Const":        ["add", "sub", "mul", "div", "sin", "cos", "exp", "log", 1.0, 5.0, 10.0],
        "Koza-Sign-Const":   ["add", "sub", "mul", "div", "sin", "cos", "exp", "log", "sign", 1.0, 5.0, 10.0],
        "Koza-Chaotic-Const":["add", "sub", "mul", "div", "sin", "cos", "exp", "log", "tanh","cosh", "abs", "sign", 1.0, 5.0, 10.0],
        "Koza-Turb-Const":   ["add", "sub", "mul", "div", "sin", "cos", "exp", "log", "tanh","cosh", 1.0, 5.0, 10.0],
    }
    
    return {
        "task": {
            "task_type": "regression",
            "function_set": function_set_dict[function_set],
        },
        "training": {
            "n_samples": 2000000,
            "batch_size": 500,
            "epsilon": 0.02,
            "n_cores_batch": 1,
        },
        "controller": {
            # "pqt": False,  # False,
            "learning_rate": 0.0025,
            "entropy_weight": 0.03,
            "entropy_gamma": 0.7,
        },
        "prior": {
            "length": {
                "min_": 4,
                # "max_": 256,
                "max_": 30,
                # "max_": 50,
                "on": True,
            },
            # Memory sanity value. Have at most 10 optimizable constants.
            # This can be set very high, but it runs rather slow.
            "repeat": {"tokens": "const", "min_": None, "max_": 10, "on": True},
            "inverse": {"on": True},
            "trig": {"on": True},
            "const": {"on": True},
            "no_inputs": {"on": True},
            "uniform_arity": {"on": False},
            "soft_length": {"loc": 10, "scale": 5, "on": True},
        },
    }


def nesymres_dataset_config_factory():
    return {
        "max_len": 20,
        "operators": "add:10,mul:10,sub:5,div:5,pow2:4,pow3:2,pow4:1,pow5:1,ln:4,exp:4,sin:4,cos:4",  # Koza
        "max_ops": 5,
        "rewrite_functions": "",
        "variables": ["x_1", "x_2", "x_3"],
        "eos_index": 1,
        "pad_index": 0,
        "equal_prob_independent_vars": True,
        "remap_independent_vars_to_monotic": True,
        "force_all_independent_present": True,
        "max_independent_vars": 16,
        "lower_nbs_ops": 3,
        "create_eqs_with_constants": True,
    }


def nesymres_train_config_factory():
    return OmegaConf.create(
        {
            "train_path": "data/datasets/100000",
            "val_path": "data/raw_datasets/200",
            "raw_test_path": "data/raw_datasets/200",
            "test_path": "data/validation",
            "model_path": "/local/home/lbiggio/NeuralSymbolicRegressionThatScales/weights/10MCompleted.ckpt",
            "wandb": False,
            "num_of_workers": 24,  # 24,
            "batch_size": 150,
            "epochs": 20,
            "val_check_interval": 1.0,
            "precision": 16,
            "gpu": 1,
            "dataset_train": {
                "total_variables": None,
                "total_coefficients": None,
                "max_number_of_points": 20,
                "type_of_sampling_points": "constant",  # 'logarithm',
                "predict_c": True,
                "fun_support": {"max": 1, "min": -1},
                "constants": {
                    "num_constants": 3,
                    "additive": {"max": 2, "min": -2},
                    "multiplicative": {"max": 2, "min": -2},
                },
            },
            "dataset_val": {
                "total_variables": None,
                "total_coefficients": None,
                "max_number_of_points": 20,
                "type_of_sampling_points": "constant",
                "predict_c": True,
                "fun_support": {"max": 1, "min": -1},
                "constants": {
                    "num_constants": 3,
                    "additive": {"max": 2, "min": -2},
                    "multiplicative": {"max": 5, "min": 0.1},
                },
            },
            "dataset_test": {
                "total_variables": None,
                "total_coefficients": None,
                "max_number_of_points": 20,
                "type_of_sampling_points": "constant",
                "predict_c": True,
                "fun_support": {"max": 1, "min": -1},
                "constants": {
                    "num_constants": 3,
                    "additive": {"max": 2, "min": -2},
                    "multiplicative": {"max": 5, "min": 0.1},
                },
            },
            "architecture": {
                "sinuisodal_embeddings": False,
                "dec_pf_dim": 32,
                "dec_layers": 1,
                "dim_hidden": 32,
                "lr": 0.0001,
                "dropout": 0,
                "num_features": 2,
                "ln": True,
                "N_p": 0,
                "num_inds": 50,
                "activation": "relu",
                "bit16": True,
                "norm": True,
                "linear": False,
                "input_normalization": False,
                "src_pad_idx": 0,
                "trg_pad_idx": 0,
                "length_eq": 20,
                "n_l_enc": 5,
                "mean": 0.5,
                "std": 0.5,
                "dim_input": 4,
                "num_heads": 2,
                "output_dim": 10,
            },
            "inference": {
                "beam_size": 2,
                "bfgs": {
                    "activated": True,
                    "n_restarts": 10,
                    "add_coefficients_if_not_existing": False,
                    "normalization_o": False,
                    "idx_remove": True,
                    "normalization_type": "MSE",
                    "stop_time": 1000000000.0,
                },
            },
        }
    )


def nesymres_function_set_factory():
    return [
        "abs",
        "arccos",
        "add",
        "arcsin",
        "arctan",
        "cos",
        "cosh",
        "coth",
        "div",
        "exp",
        "log",
        "mul",
        "sin",
        "sinh",
        "sqrt",
        "tan",
        "tanh",
        "inv",
        "neg",
        "-3",
        "-2",
        "-1",
        "0",
        "1",
        "2",
        "3",
        "4",
        "5",
    ]


def flatten(d, parent_key="", sep="_"):
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, collections.MutableMapping):
            items.extend(flatten(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def get_config(skip_cli=True):
    base_conf = OmegaConf.load("config.yaml")
    if skip_cli:
        return base_conf
    flat_base_conf = flatten(base_conf)
    cli_conf = OmegaConf.from_cli()
    cli_conf = OmegaConf.create({(k[2:] if k[:2] == "--" else k): v for k, v in cli_conf.items()})  # pyright: ignore
    flat_cli_conf = flatten(cli_conf)

    list_cond = [k in flat_base_conf for k in flat_cli_conf.keys()]
    contains_all_keys_bool = all(list_cond)
    assert contains_all_keys_bool, f"Input CLI keys that cannot be set {set(flat_cli_conf) - set(flat_base_conf)}"
    conf = OmegaConf.merge(base_conf, cli_conf)
    return conf


if __name__ == "__main__":
    conf = get_config()
    print("priority_queue_training: ", conf.experiment.priority_queue_training)
    print("seed_runs: ", conf.experiment.seed_runs)
    print("")
