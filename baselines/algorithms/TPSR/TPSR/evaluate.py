import numpy as np
import pandas as pd
import os
from collections import OrderedDict, defaultdict
from symbolicregression.model.sklearn_wrapper import SymbolicTransformerRegressor , get_top_k_features
from symbolicregression.model.model_wrapper import ModelWrapper
import symbolicregression.model.utils_wrapper as utils_wrapper

from symbolicregression.metrics import compute_metrics
from sklearn.model_selection import train_test_split
from tpsr import tpsr_fit
import time
from tqdm import tqdm
import copy


def read_file(filename, label="target", sep=None): 
    if filename.endswith("gz"):
        compression = "gzip"
    else:
        compression = None
    if sep:
        input_data = pd.read_csv(filename, sep=sep, compression=compression)
    else:
        input_data = pd.read_csv(
            filename, sep=sep, compression=compression, engine="python"
        )
    feature_names = [x for x in input_data.columns.values if x != label]
    feature_names = np.array(feature_names)
    X = input_data.drop(label, axis=1).values.astype(float)
    y = input_data[label].values
    assert X.shape[1] == feature_names.shape[0]

    return X, y, feature_names


def evaluate_pmlb(
        trainer,
        params,
        model,
        target_noise=0.0,
        random_state=29910,
        verbose=False,
        save=True,
        filter_fn=None,
        logger=None,
        save_file=None,
        save_suffix="./eval_result/eval_pmlb_feynman_pretrained.csv",
    ):
        scores = defaultdict(list)
        env = trainer.env
        params = params
        embedder = model.embedder
        encoder = model.encoder
        decoder = model.decoder
        embedder.eval()
        encoder.eval()
        decoder.eval()

        mw = ModelWrapper(
            env=env,
            embedder=embedder,
            encoder=encoder,
            decoder=decoder,
            beam_length_penalty=params.beam_length_penalty,
            beam_size=params.beam_size,
            max_generated_output_len=params.max_generated_output_len,
            beam_early_stopping=params.beam_early_stopping,
            beam_temperature=params.beam_temperature,
            beam_type=params.beam_type,
        )

        dstr = SymbolicTransformerRegressor(
            model=mw,
            max_input_points=params.max_input_points,
            n_trees_to_refine=params.n_trees_to_refine,
            max_number_bags=params.max_number_bags,
            rescale=params.rescale,
        )

        all_datasets = pd.read_csv(
            "./datasets/pmlb/pmlb/all_summary_stats.tsv",
            sep="\t",
        )
        regression_datasets = all_datasets[all_datasets["task"] == "regression"]
        regression_datasets = regression_datasets[
            regression_datasets["n_categorical_features"] == 0
        ]
        problems = regression_datasets

        if filter_fn is not None:
            problems = problems[filter_fn(problems)]
        problem_names = problems["dataset"].values.tolist()
        pmlb_path = "./datasets/pmlb/datasets/"

        feynman_problems = pd.read_csv(
            "./datasets/feynman/FeynmanEquations.csv",
            delimiter=",",
        )
        feynman_problems = feynman_problems[["Filename", "Formula"]].dropna().values
        feynman_formulas = {}
        for p in range(feynman_problems.shape[0]):
            feynman_formulas[
                "feynman_" + feynman_problems[p][0].replace(".", "_")
            ] = feynman_problems[p][1]

        first_write = True
        if save:
            save_file = save_suffix

        rng = np.random.RandomState(random_state)
        pbar = tqdm(total=len(problem_names))
        for problem_name in problem_names:
            if problem_name in feynman_formulas:
                formula = feynman_formulas[problem_name]
            else:
                formula = "???"
            print("formula : ", formula)

            X, y, _ = read_file(
                pmlb_path + "{}/{}.tsv.gz".format(problem_name, problem_name)
            )
            y = np.expand_dims(y, -1)

            x_to_fit, x_to_predict, y_to_fit, y_to_predict = train_test_split(
                X, y, test_size=0.25, shuffle=True, random_state=random_state
            )

            scale = target_noise * np.sqrt(np.mean(np.square(y_to_fit)))
            noise = rng.normal(loc=0.0, scale=scale, size=y_to_fit.shape)
            y_to_fit += noise

            dstr.fit(x_to_fit, y_to_fit, verbose=verbose)
            problem_results = defaultdict(list)
           
            for refinement_type in dstr.retrieve_refinements_types():
                best_gen = copy.deepcopy(
                    dstr.retrieve_tree(refinement_type=refinement_type, with_infos=True)
                )
                predicted_tree = best_gen["predicted_tree"]
                if predicted_tree is None:
                    continue
                del best_gen["predicted_tree"]
                if "metrics" in best_gen:
                    del best_gen["metrics"]

                problem_results["predicted_tree"].append(predicted_tree)
                problem_results["predicted_tree_prefix"].append(
                    predicted_tree.prefix() if predicted_tree is not None else None
                )
                for info, val in best_gen.items():
                    problem_results[info].append(val)

                y_tilde_to_fit = dstr.predict(x_to_fit, refinement_type=refinement_type)
                results_fit = compute_metrics(
                    {
                        "true": [y_to_fit],
                        "predicted": [y_tilde_to_fit],
                        "predicted_tree": [predicted_tree],
                    },
                    metrics=params.validation_metrics,
                )
                for k, v in results_fit.items():
                    problem_results[k + "_fit"].extend(v)
                    scores[refinement_type + "|" + k + "_fit"].extend(v)

                y_tilde_to_predict = dstr.predict(
                    x_to_predict, refinement_type=refinement_type
                )
                results_predict = compute_metrics(
                    {
                        "true": [y_to_predict],
                        "predicted": [y_tilde_to_predict],
                        "predicted_tree": [predicted_tree],
                    },
                    metrics=params.validation_metrics,
                )
                for k, v in results_predict.items():
                    problem_results[k + "_predict"].extend(v)
                    scores[refinement_type + "|" + k + "_predict"].extend(v)

            problem_results = pd.DataFrame.from_dict(problem_results)
            problem_results.insert(0, "problem", problem_name)
            problem_results.insert(0, "formula", formula)
            problem_results["input_dimension"] = x_to_fit.shape[1]

            if save:
                if first_write:
                    problem_results.to_csv(save_file, index=False)
                    first_write = False
                else:
                    problem_results.to_csv(
                        save_file, mode="a", header=False, index=False
                    )
            pbar.update(1)
        for k, v in scores.items():
            scores[k] = np.nanmean(v)
        return scores


def evaluate_pmlb_mcts(
        trainer,
        params,
        model,
        target_noise=0.0,
        random_state=29910,
        verbose=False,
        save=True,
        filter_fn=None,
        logger=None,
        save_file=None,
        save_suffix="./eval_result/eval_pmlb_tpsr.csv",
        rescale = True
    ):
        scores = defaultdict(list)
        env = trainer.env
        params = params
        embedder = model.embedder
        encoder = model.encoder
        decoder = model.decoder
        embedder.eval()
        encoder.eval()
        decoder.eval()

        mw = ModelWrapper(
            env=env,
            embedder=embedder,
            encoder=encoder,
            decoder=decoder,
            beam_length_penalty=params.beam_length_penalty,
            beam_size=params.beam_size,
            max_generated_output_len=params.max_generated_output_len,
            beam_early_stopping=params.beam_early_stopping,
            beam_temperature=params.beam_temperature,
            beam_type=params.beam_type,
        )

        dstr = SymbolicTransformerRegressor(
            model=mw,
            max_input_points=params.max_input_points,
            n_trees_to_refine=params.n_trees_to_refine,
            max_number_bags=params.max_number_bags,
            rescale=params.rescale,
        )

        problem_names = os.listdir("data/transformed")
        first_write = True
        if save:
            save_file = save_suffix

        rng = np.random.RandomState(random_state)
        pbar = tqdm(total=len(problem_names))
        
        for problem_name in problem_names:
                
            formula = "???"
            data = np.loadtxt(f"data/transformed/{problem_name}", delimiter=',')
            X = data[:,:-1]
            y = np.expand_dims(data[:,-1], -1)

            x_to_fit, x_to_predict, y_to_fit, y_to_predict = train_test_split(
                X, y, test_size=0.25, shuffle=True, random_state=random_state
            )

            scale = target_noise * np.sqrt(np.mean(np.square(y_to_fit)))
            noise = rng.normal(loc=0.0, scale=scale, size=y_to_fit.shape)
            y_to_fit += noise
            
            ## Scale X 
            if not isinstance(X, list):
                X = [x_to_fit]
                Y = [y_to_fit]
            n_datasets = len(X)
            dstr.top_k_features = [None for _ in range(n_datasets)]
            for i in range(n_datasets):
                dstr.top_k_features[i] = get_top_k_features(X[i], Y[i], k=dstr.model.env.params.max_input_dimension)
                X[i] = X[i][:, dstr.top_k_features[i]]

            scaler = utils_wrapper.StandardScaler() if rescale else None
            scale_params = {}
            if scaler is not None:
                scaled_X = []
                for i, x in enumerate(X):
                    scaled_X.append(scaler.fit_transform(x))
                    scale_params[i]=scaler.get_params()
            else:
                scaled_X = X

            bag_number =1 
            done_bagging = False
            bagging_threshold = 0.99 
            max_r2_zero = 0
            max_bags = min(11,len(scaled_X[0])//params.max_input_points+2)
            while done_bagging == False and bag_number<max_bags:
                s, time_elapsed, sample_times = tpsr_fit(model, scaled_X, Y, params,env , bag_number)
                print("time elapsed for sample: ", time_elapsed)
                generated_tree = list(filter(lambda x: x is not None,
                            [env.idx_to_infix(s[1:], is_float=False, str_array=False)]))
                if generated_tree == []: 
                    y = None
                    model_str= None
                    tree = None
                else:
                    dstr.start_fit = time.time()
                    dstr.tree = {}
                    refined_candidate = dstr.refine(scaled_X[0], Y[0], generated_tree, verbose=False)
                    if scaler is not None:
                        refined_candidate[0]["predicted_tree"]=scaler.rescale_function(env, refined_candidate[0]["predicted_tree"], *scale_params[0])
                    dstr.tree[0] = refined_candidate
                    
                problem_results = defaultdict(list)
            
                for refinement_type in dstr.retrieve_refinements_types():
                    best_gen = copy.deepcopy(
                        dstr.retrieve_tree(refinement_type=refinement_type, with_infos=True)
                    )
                    predicted_tree = best_gen["predicted_tree"]
                    if predicted_tree is None:
                        continue
                    del best_gen["predicted_tree"]
                    if "metrics" in best_gen:
                        del best_gen["metrics"]

                    problem_results["predicted_tree"].append(predicted_tree)
                    problem_results["predicted_tree_prefix"].append(
                        predicted_tree.prefix() if predicted_tree is not None else None
                    )
                    for info, val in best_gen.items():
                        problem_results[info].append(val)

                    y_tilde_to_fit = dstr.predict(x_to_fit, refinement_type=refinement_type)
                    results_fit = compute_metrics(
                        {
                            "true": [y_to_fit],
                            "predicted": [y_tilde_to_fit],
                            "predicted_tree": [predicted_tree],
                        },
                        metrics=params.validation_metrics,
                    )
                    for k, v in results_fit.items():
                        problem_results[k + "_fit"].extend(v)
                        scores[refinement_type + "|" + k + "_fit"].extend(v)

                    y_tilde_to_predict = dstr.predict(
                        x_to_predict, refinement_type=refinement_type
                    )
                    results_predict = compute_metrics(
                        {
                            "true": [y_to_predict],
                            "predicted": [y_tilde_to_predict],
                            "predicted_tree": [predicted_tree],
                        },
                        metrics=params.validation_metrics,
                    )
                    for k, v in results_predict.items():
                        if k == "r2_zero" and refinement_type == "BFGS":
                            if v[0] > bagging_threshold:
                                done_bagging = True
                            else:
                                bag_number += 1
                        problem_results[k + "_predict"].extend(v)
                        scores[refinement_type + "|" + k + "_predict"].extend(v)

                problem_results = pd.DataFrame.from_dict(problem_results)
                problem_results.insert(0, "problem", problem_name)
                problem_results.insert(0, "formula", formula)
                problem_results["input_dimension"] = x_to_fit.shape[1]
                problem_results["time"] = time_elapsed
                problem_results["sample_times"] = sample_times
                problem_results["bag_number"] = bag_number
                if problem_results["r2_zero_fit"][0] > max_r2_zero:
                    final_results = problem_results.copy()
                    max_r2_zero = problem_results["r2_zero_fit"][0]
            
            if save:
                if first_write:
                    final_results.to_csv(save_file, index=False)
                    first_write = False
                else:
                    final_results.to_csv(
                        save_file, mode="a", header=False, index=False
                    )
            pbar.update(1)
        for k, v in scores.items():
            scores[k] = np.nanmean(v)
            
        return scores


def evaluate_in_domain(
        trainer,
        params,
        model,
        data_type,
        task,
        verbose=True,
        ablation_to_keep=None,
        save=False,
        logger=None,
        save_file=None,
    ):

        """
        Encoding / decoding step with beam generation and SymPy check.
        """
        scores = OrderedDict({"epoch": trainer.epoch})

        params = params
        embedder =model.embedder
        encoder = model.encoder
        decoder = model.decoder
        embedder.eval()
        encoder.eval()
        decoder.eval()

        env = trainer.env

        eval_size_per_gpu = params.eval_size #old
        iterator = env.create_test_iterator(
            data_type,
            task,
            data_path=trainer.data_path,
            batch_size=params.batch_size_eval,
            params=params,
            size=eval_size_per_gpu,
            input_length_modulo=params.eval_input_length_modulo,
            test_env_seed=params.test_env_seed,
        )

        mw = ModelWrapper(
            env=env,
            embedder=embedder,
            encoder=encoder,
            decoder=decoder,
            beam_length_penalty=params.beam_length_penalty,
            beam_size=params.beam_size,
            max_generated_output_len=params.max_generated_output_len,
            beam_early_stopping=params.beam_early_stopping,
            beam_temperature=params.beam_temperature,
            beam_type=params.beam_type,
        )

        dstr = SymbolicTransformerRegressor(
            model=mw,
            max_input_points=params.max_input_points,
            n_trees_to_refine=params.n_trees_to_refine,
            rescale=False,
        )

        first_write = True
        if save:
            if save_file is None:
                save_file = (
                    params.eval_dump_path
                    if params.eval_dump_path is not None
                    else params.dump_path
                )
            if not os.path.exists(save_file):
                os.makedirs(save_file)
            save_file = os.path.join(save_file, "eval_in_domain_pretrained_{}.csv".format(params.beam_type))

        batch_before_writing_threshold = min(
            2, eval_size_per_gpu // params.batch_size_eval
        )
        batch_before_writing = batch_before_writing_threshold

        if ablation_to_keep is not None:
            ablation_to_keep = list(
                map(lambda x: "info_" + x, ablation_to_keep.split(","))
            )
        else:
            ablation_to_keep = []

        pbar = tqdm(total=eval_size_per_gpu)
        batch_results = defaultdict(list)

        for samples, _ in iterator:
            x_to_fit = samples["x_to_fit"]
            y_to_fit = samples["y_to_fit"]
            infos = samples["infos"]
            tree = samples["tree"]
            
            dstr.fit(x_to_fit, y_to_fit, verbose=verbose)
            for k, v in infos.items():
                infos[k] = v.tolist()

            for refinement_type in dstr.retrieve_refinements_types():

                best_gens = copy.deepcopy(
                    dstr.retrieve_tree(
                        refinement_type=refinement_type, dataset_idx=-1, with_infos=True
                    )
                )
                predicted_tree = [best_gen["predicted_tree"] for best_gen in best_gens]
                for best_gen in best_gens:
                    del best_gen["predicted_tree"]
                    if "metrics" in best_gen:
                        del best_gen["metrics"]

                batch_results["predicted_tree"].extend(predicted_tree)
                batch_results["predicted_tree_prefix"].extend(
                    [
                        _tree.prefix() if _tree is not None else np.NaN
                        for _tree in predicted_tree
                    ]
                )
                for best_gen in best_gens:
                    for info, val in best_gen.items():
                        batch_results[info].extend([val])

                for k, v in infos.items():
                    batch_results["info_" + k].extend(v)

                y_tilde_to_fit = dstr.predict(
                    x_to_fit, refinement_type=refinement_type, batch=True
                )
                assert len(y_to_fit) == len(
                    y_tilde_to_fit
                ), "issue with len, tree: {}, x:{} true: {}, predicted: {}".format(
                    len(predicted_tree),
                    len(x_to_fit),
                    len(y_to_fit),
                    len(y_tilde_to_fit),
                )
                results_fit = compute_metrics(
                    {
                        "true": y_to_fit,
                        "predicted": y_tilde_to_fit,
                        "tree": tree,
                        "predicted_tree": predicted_tree,
                    },
                    metrics=params.validation_metrics,
                )
                for k, v in results_fit.items():
                    batch_results[k + "_fit"].extend(v)
                del results_fit

                if params.prediction_sigmas is None:
                    prediction_sigmas = []
                else:
                    prediction_sigmas = [
                        float(sigma)
                        for sigma in params.prediction_sigmas.split(",")
                    ]
                    
                for sigma in prediction_sigmas:
                    x_to_predict = samples["x_to_predict_{}".format(sigma)]
                    y_to_predict = samples["y_to_predict_{}".format(sigma)]
                    y_tilde_to_predict = dstr.predict(
                        x_to_predict, refinement_type=refinement_type, batch=True
                    )
                    results_predict = compute_metrics(
                        {
                            "true": y_to_predict,
                            "predicted": y_tilde_to_predict,
                            "tree": tree,
                            "predicted_tree": predicted_tree,
                        },
                        metrics=params.validation_metrics,
                    )
                    for k, v in results_predict.items():
                        batch_results[k + "_predict_{}".format(sigma)].extend(v)
                    del results_predict

                batch_results["tree"].extend(tree)
                batch_results["tree_prefix"].extend([_tree.prefix() for _tree in tree])
                
            if save:

                batch_before_writing -= 1
                if batch_before_writing <= 0:
                    batch_results = pd.DataFrame.from_dict(batch_results)
                    if first_write:
                        batch_results.to_csv(save_file, index=False)
                        first_write = False
                    else:
                        batch_results.to_csv(
                            save_file, mode="a", header=False, index=False
                        )
                    batch_before_writing = batch_before_writing_threshold
                    batch_results = defaultdict(list)

            bs = len(x_to_fit)
            pbar.update(bs)

        try:
            df = pd.read_csv(save_file, na_filter=True)
        except:
            return
        info_columns = filter(lambda x: x.startswith("info_"), df.columns)
        df = df.drop(columns=filter(lambda x: x not in ablation_to_keep, info_columns))

        for refinement_type, df_refinement_type in df.groupby("refinement_type"):
            avg_scores = df_refinement_type.mean().to_dict()
            for k, v in avg_scores.items():
                scores[refinement_type + "|" + k] = v
            for ablation in ablation_to_keep:
                for val, df_ablation in df_refinement_type.groupby(ablation):
                    avg_scores_ablation = df_ablation.mean()
                    for k, v in avg_scores_ablation.items():
                        scores[
                            refinement_type + "|" + k + "_{}_{}".format(ablation, val)
                        ] = v
        return scores


def evaluate_mcts_in_domain(
        trainer,
        params,
        model,
        data_type,
        task,
        verbose=True,
        ablation_to_keep=None,
        save=False,
        logger=None,
        save_file=None,
        rescale = False,
        save_suffix=None
    ):

        """
        Encoding / decoding step with beam generation and SymPy check.
        """
        scores = OrderedDict({"epoch": trainer.epoch})

        params = params
        embedder =model.embedder
        encoder = model.encoder
        decoder = model.decoder
        embedder.eval()
        encoder.eval()
        decoder.eval()

        env = trainer.env

        eval_size_per_gpu = params.eval_size #old
        iterator = env.create_test_iterator(
            data_type,
            task,
            data_path=trainer.data_path,
            batch_size=params.batch_size_eval,
            params=params,
            size=eval_size_per_gpu,
            input_length_modulo=params.eval_input_length_modulo,
            test_env_seed=params.test_env_seed,
        )
        

        mw = ModelWrapper(
            env=env,
            embedder=embedder,
            encoder=encoder,
            decoder=decoder,
            beam_length_penalty=params.beam_length_penalty,
            beam_size=params.beam_size,
            max_generated_output_len=params.max_generated_output_len,
            beam_early_stopping=params.beam_early_stopping,
            beam_temperature=params.beam_temperature,
            beam_type=params.beam_type,
        )

        dstr = SymbolicTransformerRegressor(
            model=mw,
            max_input_points=params.max_input_points,
            n_trees_to_refine=params.n_trees_to_refine,
            rescale=False,
        )

        first_write = True
        if save:
            save_file = save_suffix

        batch_before_writing_threshold = min(
            2, eval_size_per_gpu // params.batch_size_eval
        )
        batch_before_writing = batch_before_writing_threshold

        if ablation_to_keep is not None:
            ablation_to_keep = list(
                map(lambda x: "info_" + x, ablation_to_keep.split(","))
            )
        else:
            ablation_to_keep = []

        pbar = tqdm(total=eval_size_per_gpu)

        batch_results = defaultdict(list)

        for samples, _ in iterator:
            x_to_fit = samples["x_to_fit"]
            y_to_fit = samples["y_to_fit"]
            infos = samples["infos"]
            tree = samples["tree"]
                        
            #### Scale X 
            X = x_to_fit
            Y = y_to_fit
            if not isinstance(X, list):
                X = [X]
                Y = [Y]
            n_datasets = len(X)
            dstr.top_k_features = [None for _ in range(n_datasets)]
            for i in range(n_datasets):
                dstr.top_k_features[i] = get_top_k_features(X[i], Y[i], k=dstr.model.env.params.max_input_dimension)
                X[i] = X[i][:, dstr.top_k_features[i]]

            scaler = utils_wrapper.StandardScaler() if rescale else None
            scale_params = {}
            if scaler is not None:
                scaled_X = []
                for i, x in enumerate(X):
                    scaled_X.append(scaler.fit_transform(x))
                    scale_params[i]=scaler.get_params()
            else:
                scaled_X = X

            s, time_elapsed, sample_times = tpsr_fit(scaled_X, Y, params,env)
            print("time elapsed for sample: ", time_elapsed)
            replace_ops = {"add": "+", "mul": "*", "sub": "-", "pow": "**", "inv": "1/"}
            generated_tree = list(filter(lambda x: x is not None,
                        [env.idx_to_infix(s[1:], is_float=False, str_array=False)]))
            if generated_tree == []: 
                y = None
                model_str= None
                tree = None
            else:
                dstr.start_fit = time.time()
                dstr.tree = {}             
                refined_candidate = dstr.refine(scaled_X[0], Y[0], generated_tree, verbose=False)
                dstr.tree[0] = refined_candidate

            for k, v in infos.items():
                infos[k] = v.tolist()
                
            for refinement_type in dstr.retrieve_refinements_types():

                best_gen = copy.deepcopy(
                        dstr.retrieve_tree(refinement_type=refinement_type, with_infos=True)
                    )
                predicted_tree = best_gen["predicted_tree"]
                if predicted_tree is None:
                    continue
                del best_gen["predicted_tree"]
                if "metrics" in best_gen:
                    del best_gen["metrics"]

                batch_results["predicted_tree"].append(predicted_tree)
                batch_results["predicted_tree_prefix"].append(
                    predicted_tree.prefix() if predicted_tree is not None else None
                )
                for info, val in best_gen.items():
                    batch_results[info].append(val)
                    
                for k, v in infos.items():
                    batch_results["info_" + k].extend(v)
                        
                y_tilde_to_fit = dstr.predict(x_to_fit, refinement_type=refinement_type)
                
                results_fit = compute_metrics(
                    {
                        "true": y_to_fit,
                        "predicted": [y_tilde_to_fit],
                        "tree": tree,
                        "predicted_tree": [predicted_tree],
                    },
                    metrics=params.validation_metrics,
                )
                for k, v in results_fit.items():
                    batch_results[k + "_fit"].extend(v)
                del results_fit

                if params.prediction_sigmas is None:
                    prediction_sigmas = []
                else:
                    prediction_sigmas = [
                        float(sigma)
                        for sigma in params.prediction_sigmas.split(",")
                    ]

                for sigma in prediction_sigmas:
                    x_to_predict = samples["x_to_predict_{}".format(sigma)]
                    y_to_predict = samples["y_to_predict_{}".format(sigma)]
                    y_tilde_to_predict = dstr.predict(
                        x_to_predict, refinement_type=refinement_type
                    )
                    results_predict = compute_metrics(
                        {
                            "true": y_to_predict,
                            "predicted": [y_tilde_to_predict],
                            "tree": tree,
                            "predicted_tree": [predicted_tree],
                        },
                        metrics=params.validation_metrics,
                    )
                    for k, v in results_predict.items():
                        batch_results[k + "_predict_{}".format(sigma)].extend(v)
                    del results_predict

                batch_results["tree"].extend(tree)
                batch_results["tree_prefix"].extend([_tree.prefix() for _tree in tree])
                batch_results["time_mcts"].extend([time_elapsed])
                batch_results["sample_times"].extend([sample_times])
                
            if save:

                batch_before_writing -= 1
                if batch_before_writing <= 0:
                    batch_results = pd.DataFrame.from_dict(batch_results)
                    if first_write:
                        batch_results.to_csv(save_file, index=False)
                        first_write = False
                    else:
                        batch_results.to_csv(
                            save_file, mode="a", header=False, index=False
                        )
                    batch_before_writing = batch_before_writing_threshold
                    batch_results = defaultdict(list)

            bs = len(x_to_fit)
            pbar.update(bs)

        try:
            df = pd.read_csv(save_file, na_filter=True)
        except:
            return
        info_columns = filter(lambda x: x.startswith("info_"), df.columns)
        df = df.drop(columns=filter(lambda x: x not in ablation_to_keep, info_columns))

        for refinement_type, df_refinement_type in df.groupby("refinement_type"):
            avg_scores = df_refinement_type.mean().to_dict()
            for k, v in avg_scores.items():
                scores[refinement_type + "|" + k] = v
            for ablation in ablation_to_keep:
                for val, df_ablation in df_refinement_type.groupby(ablation):
                    avg_scores_ablation = df_ablation.mean()
                    for k, v in avg_scores_ablation.items():
                        scores[
                            refinement_type + "|" + k + "_{}_{}".format(ablation, val)
                        ] = v
        return scores
