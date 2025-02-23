import os
import torch
import numpy as np
import symbolicregression
import requests

from symbolicregression.envs import build_env
from symbolicregression.model import build_modules
from parsers import get_parser
from symbolicregression.trainer import Trainer
from symbolicregression.slurm import init_signal_handler, init_distributed_mode
from evaluate import evaluate_pmlb, evaluate_pmlb_mcts, evaluate_in_domain, evaluate_mcts_in_domain



if __name__ == '__main__':
    #load E2E model
    
    model_path = "periodic-0.pth" 
    try:
        if not os.path.isfile(model_path): 
            url = "https://dl.fbaipublicfiles.com/symbolicregression/model1.pt"
            r = requests.get(url, allow_redirects=True)
            open(model_path, 'wb').write(r.content)
        if not torch.cuda.is_available():
            model = torch.load(model_path, map_location=torch.device('cpu'))
        else:
            model = torch.load(model_path)
        if isinstance(model, dict):
            class Model:
                def __init__(self, model):
                    self.embedder = model['embedder']
                    self.encoder = model['encoder']
                    self.decoder = model['decoder']
            model = Model(model)
        print("Model successfully loaded!") 
        print(model.embedder)

    except Exception as e:
        print("ERROR: model not loaded! path was: {}".format(model_path))
        print(e)
    

    #load data:
    parser = get_parser()
    params = parser.parse_args()
    init_distributed_mode(params)
    if params.is_slurm_job:
        init_signal_handler()

    np.random.seed(params.seed)
    torch.manual_seed(params.seed)
    torch.cuda.manual_seed(params.seed)
    
    # CPU / CUDA
    if not params.cpu:
        assert torch.cuda.is_available()
    params.eval_only = True
    symbolicregression.utils.CUDA = not params.cpu
     
    env = build_env(params)
    env.rng = np.random.RandomState(0)
    modules = build_modules(env, params)
    trainer = Trainer(modules, env, params)
    
    params.batch_size_eval = 1
    params.multi_gpu = False
    params.is_slurm_job = False
    params.random_state = 14423
    params.eval_verbose_print = True
    params.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(params.no_prefix_cache, params.no_seq_cache)
    if (params.no_prefix_cache==False) and (params.no_seq_cache == False):
        params.cache = 'b' #both cachings are arctivated
    elif (params.no_prefix_cache==True) and (params.no_seq_cache ==False):
        params.cache = 's' #only sequence caching
    elif (params.no_prefix_cache==False) and (params.no_seq_cache==True):
        params.cache = 'k' #only top-k caching
    else:
        params.cache = 'n' # no caching

    target_noise = params.target_noise
    random_state = params.random_state
    data_type = params.pmlb_data_type
    save = params.save_results

    if data_type == "feynman":
        filter_fn = lambda x: x["dataset"].str.contains("feynman")
    elif data_type == "strogatz":
        print("Strogatz data")
        filter_fn = lambda x: x["dataset"].str.contains("strogatz")
    elif data_type == "603_fri_c0_250_50":
        filter_fn = lambda x: x["dataset"].str.contains("603_fri_c0_250_50")
    else:
        filter_fn = lambda x: ~(
            x["dataset"].str.contains("strogatz")
            | x["dataset"].str.contains("feynman")
        )

    pmlb_scores = evaluate_pmlb_mcts(
        trainer,
        params,
        model,
        target_noise=target_noise,
        verbose=params.eval_verbose_print,
        random_state=random_state,
        save=save,
        filter_fn=filter_fn,
        save_file=None,
        save_suffix="./eval_result/eval_{}_tpsr_l{}_b{}_k{}_r{}_cache{}_noise{}.csv".format(params.pmlb_data_type,
                                                                                                params.lam,
                                                                                                params.num_beams,
                                                                                                params.width,
                                                                                                params.rollout,
                                                                                                params.cache,
                                                                                                params.target_noise),
    )
    print("TPSR scores: ", pmlb_scores)
      
