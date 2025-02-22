from functools import partial
from pathlib import Path

import torch
from nesymres.scripts.data_creation.dataset_creation import creator
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def generate_train_and_val_functions(
    function_names,
    function_set_name,
    nesymres_train_config,
    nesymres_dataset_config,
    training_equations=200,
    train_path=None,
    return_true_eq=False,
    train_global_seed=0,
):
    from nesymres.architectures.data import NesymresDataset, custom_collate_fn

    if train_path == "":
        train_path = creator(
            config=nesymres_dataset_config,
            number_of_equations=training_equations,
            ds_key="{}-{}".format(",".join(function_names), nesymres_dataset_config["variables"][-1]), # kk here control n var pretrain
            global_seed=train_global_seed,
        )
    print("Generating val set 100 equations")
    val_path = creator(
        config=nesymres_dataset_config,
        number_of_equations=100,
        ds_key="{}-{}".format(",".join(function_names), nesymres_dataset_config["variables"][-1]),
        global_seed=9999999,
    )
    dltrain = DataLoader(
        NesymresDataset(
            train_path,  # pyright: ignore
            nesymres_train_config.dataset_train,
            mode="train",
            return_true_eq=return_true_eq,
        ),
        batch_size=nesymres_train_config.batch_size,
        shuffle=False,
        drop_last=True,
        collate_fn=partial(custom_collate_fn, cfg=nesymres_train_config.dataset_train, return_true_eq=return_true_eq),
        num_workers=nesymres_train_config.num_of_workers,
        pin_memory=True,
    )
    dlval = DataLoader(
        NesymresDataset(val_path, nesymres_train_config.dataset_train, mode="val", return_true_eq=return_true_eq),
        batch_size=nesymres_train_config.batch_size,
        shuffle=False,
        drop_last=True,
        collate_fn=partial(custom_collate_fn, cfg=nesymres_train_config.dataset_train, return_true_eq=return_true_eq),
        num_workers=nesymres_train_config.num_of_workers,
        pin_memory=True,
    )
    return dltrain, dlval, None, train_path


def generate_pretrain_data_set(name=None, batch_size=25, normalize=True, noise_std=None, num_workers=0):
    from models.nesymres.architectures.data import NesymresDataset, custom_collate_fn

    data_train_path = Path("data/raw_datasets/200")
    data_val_path = Path("data/raw_datasets/200")

    dataset_train_cfg = OmegaConf.create(
        {
            "total_variables": ["x_1", "x_2", "x_3"],
            "total_coefficients": [
                "cm_0",
                "cm_1",
                "cm_2",
                "cm_3",
                "cm_4",
                "cm_5",
                "cm_6",
                "cm_7",
                "cm_8",
                "cm_9",
                "cm_10",
                "cm_11",
                "cm_12",
                "cm_13",
                "cm_14",
                "cm_15",
                "cm_16",
                "cm_17",
                "cm_18",
                "cm_19",
                "cm_20",
                "cm_21",
                "cm_22",
                "cm_23",
                "cm_24",
                "cm_25",
                "cm_26",
                "cm_27",
                "cm_28",
                "cm_29",
                "cm_30",
                "cm_31",
                "cm_32",
                "cm_33",
                "cm_34",
                "cm_35",
                "cm_36",
                "cm_37",
                "cm_38",
                "cm_39",
                "ca_0",
                "ca_1",
                "ca_2",
                "ca_3",
                "ca_4",
                "ca_5",
                "ca_6",
                "ca_7",
                "ca_8",
                "ca_9",
                "ca_10",
                "ca_11",
                "ca_12",
                "ca_13",
                "ca_14",
                "ca_15",
                "ca_16",
                "ca_17",
                "ca_18",
                "ca_19",
                "ca_20",
                "ca_21",
                "ca_22",
                "ca_23",
                "ca_24",
                "ca_25",
                "ca_26",
                "ca_27",
                "ca_28",
                "ca_29",
                "ca_30",
                "ca_31",
                "ca_32",
                "ca_33",
                "ca_34",
                "ca_35",
                "ca_36",
                "ca_37",
                "ca_38",
                "ca_39",
            ],
            "max_number_of_points": 500,
            "type_of_sampling_points": "constant",
            "predict_c": True,
            "fun_support": {"max": 10, "min": -10},
            "constants": {
                "num_constants": 3,
                "additive": {"max": 2, "min": -2},
                "multiplicative": {"max": 5, "min": 0.1},
            },
        }
    )

    dataset_val_cfg = OmegaConf.create(
        {
            "total_variables": None,
            "total_coefficients": None,
            "max_number_of_points": 500,
            "type_of_sampling_points": "constant",
            "predict_c": True,
            "fun_support": {"max": 10, "min": -10},
            "constants": {
                "num_constants": 3,
                "additive": {"max": 2, "min": -2},
                "multiplicative": {"max": 5, "min": 0.1},
            },
        }
    )

    train_dataset = NesymresDataset(data_train_path, dataset_train_cfg, mode="train")

    val_dataset = NesymresDataset(data_val_path, dataset_val_cfg, mode="val")

    dltrain = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        collate_fn=partial(custom_collate_fn, cfg=dataset_train_cfg),
        num_workers=num_workers,
        pin_memory=True,
    )
    dlval = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        collate_fn=partial(custom_collate_fn, cfg=dataset_train_cfg),
        num_workers=num_workers,
        pin_memory=True,
    )
    dltest = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        collate_fn=partial(custom_collate_fn, cfg=dataset_train_cfg),
        num_workers=num_workers,
        pin_memory=True,
    )

    return dltrain, dlval, dltest
