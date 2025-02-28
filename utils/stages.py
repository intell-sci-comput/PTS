import yaml


def load_yaml_config(file_path):
    with open(file_path, "r") as file:
        return yaml.safe_load(file)


def run_stages(config, reg):
    default = config["default"]
    for stage in config["stages"]:
        settings = {**default, **stage}

        flag, pareto = reg.fit_one(
            settings["operators"],
            settings["n_psrn_inputs"],
            settings["n_sample_variables"],
            settings["time_limit"],
        )
        if flag:
            return flag, pareto
    return flag, pareto
