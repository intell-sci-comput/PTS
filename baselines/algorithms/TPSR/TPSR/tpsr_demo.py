import time
import json
import os

import torch
import numpy as np
import sympy as sp
from parsers import get_parser

import symbolicregression
from symbolicregression.envs import build_env
from symbolicregression.model import build_modules
from symbolicregression.trainer import Trainer
from symbolicregression.e2e_model import Transformer, pred_for_sample_no_refine, respond_to_batch , pred_for_sample, refine_for_sample, pred_for_sample_test, refine_for_sample_test 
from dyna_gym.agents.uct import UCT
from dyna_gym.agents.mcts import update_root, convert_to_json, print_tree
from rl_env import RLEnv
from default_pi import E2EHeuristic, NesymresHeuristic
from symbolicregression.metrics import compute_metrics


from nesymres.src.nesymres.architectures.model import Model
from nesymres.utils import load_metadata_hdf5
from nesymres.dclasses import FitParams, NNEquation, BFGSParams
from functools import partial
from sympy import lambdify
from reward import compute_reward_e2e, compute_reward_nesymres
import omegaconf


def evaluate_metrics(params, y_gt, tree_gt, y_pred, tree_pred):
    metrics = [] # 7 metrics and for all samples to evaluate
    results_fit = compute_metrics(
        {
            "true": [y_gt],
            "predicted": [y_pred],
            "tree": tree_gt,
            "predicted_tree": tree_pred,
        },
        metrics=params.validation_metrics,
    )
    for k, v in results_fit.items():
        print("metric {}: ".format(k), v)
        metrics.append(v[0])
    
    return metrics

def compute_nmse(y_gt , y_pred):
    eps = 1e-9 # For avoiding Nan or Inf
    return np.sqrt( np.mean((y_gt - y_pred)**2) / (np.mean((y_gt)**2)+eps) ) 

def compute_mse(y_gt , y_pred):
    return np.mean((y_gt - y_pred)**2)



def main_e2e(case, params, equation_env, samples):
    
    model = Transformer(params = params, env=equation_env, samples=samples)
    model.to(params.device) 
    generations_ref, gen_len_ref = respond_to_batch(model, max_target_length=200, top_p=1.0, sample_temperature=None) 
    sequence_ref = generations_ref[0][:gen_len_ref-1].tolist()
    
    rl_env = RLEnv(
        params = params,
        samples = samples,
        equation_env = equation_env,
        model = model)
    
    dp = E2EHeuristic(
            equation_env=equation_env,
            rl_env=rl_env,
            model=model,
            k=params.width,
            num_beams=params.num_beams,
            horizon=params.horizon,
            device=params.device,
            use_seq_cache=not params.no_seq_cache,
            use_prefix_cache=not params.no_prefix_cache,
            length_penalty = params.beam_length_penalty,
            train_value_mode=params.train_value,
            debug=params.debug)
    
    start = time.time()
    
    agent = UCT(
        action_space=[], # this will not be used as we have a default policy
        gamma=1., # no discounting
        ucb_constant=1.,
        horizon=params.horizon,
        rollouts=params.rollout,
        dp=dp,
        width=params.width,
        reuse_tree=True,
        alg=params.uct_alg,
        ucb_base=params.ucb_base)
    
    agent.display()
    if params.sample_only:
        # a bit hacky, should set a large rollout number so all programs are saved in samples json file
        horizon = 1
    else:
        horizon = 200    
        
    done = False
    s = rl_env.state
    ret_all = []
    for t in range(horizon):
        if len(s) >= params.horizon:
            print(f'Cannot process programs longer than {params.horizon}. Stop here.')
            break
        if done:
            break
        act = agent.act(rl_env, done)
        s, r, done, _ = rl_env.step(act)
        if t ==0:
            real_root = agent.root
        if params.debug:
            # print the current tree
            print('tree:')
            # print_tree(agent.root, equation_env.equation_id2word)
            ret = convert_to_json(agent.root, rl_env, act)
            ret_all.append(ret)
            
            with open("tree.json", "w") as outfile:
                json.dump(ret_all, outfile)

            print('took action:')
            print(repr(equation_env.equation_id2word[act]))
            print('========== state (excluding prompt) ==========')
            print(s)

        update_root(agent, act, s)
        dp.update_cache(s)

    time_elapsed = time.time() - start
    y_gt = samples['y_to_fit'][0].reshape(-1)
    y_gt_test = samples['y_to_pred'][0].reshape(-1)

    y_ref , ref_str , ref_tree = pred_for_sample_no_refine(model, equation_env, sequence_ref ,samples['x_to_fit'])
    NMSE_ref = compute_nmse(y_gt, y_ref)
    ref_reward = rl_env.get_reward(sequence_ref, mode='test')
    
    y_ref_search , ref_str_search, ref_tree_search = pred_for_sample(model, equation_env,samples['x_to_fit'],samples['y_to_fit'], refine=False, beam_type='search', beam_size=100)
    NMSE_ref_search= compute_nmse(y_gt, y_ref_search)

    y_ref_sample , ref_str_sample, ref_tree_sample = pred_for_sample(model, equation_env,samples['x_to_fit'],samples['y_to_fit'], refine=False, beam_type='sampling', beam_size=100)
    NMSE_ref_sample = compute_nmse(y_gt, y_ref_sample)

    y_ref_refine , ref_str_refine, ref_tree_refine = pred_for_sample(model, equation_env,samples['x_to_fit'],samples['y_to_fit'], refine=True, beam_type='sampling', beam_size=100)
    NMSE_ref_refine = compute_nmse(y_gt, y_ref_refine)
    # MSE_ref_refine = compute_mse(y_gt, y_ref_refine)

    y_ref_refine_train ,y_ref_refine_test, _, _= pred_for_sample_test(model, equation_env,samples['x_to_fit'],samples['y_to_fit'],samples['x_to_pred'], refine=True, beam_type='sampling', beam_size=100)
    MSE_ref_refine = compute_mse(y_gt_test, y_ref_refine_test)

    y_mcts , mcts_str , mcts_tree = pred_for_sample_no_refine(model, equation_env, s ,samples['x_to_fit'])
    NMSE_mcts = compute_nmse(y_gt, y_mcts)
    final_reward = rl_env.get_reward(s, mode='test')

    y_mcts_refine , mcts_str_refine, mcts_tree_refine = refine_for_sample(params, model, equation_env, s,  samples['x_to_fit'], samples['y_to_fit'])
    NMSE_mcts_refine = compute_nmse(y_gt, y_mcts_refine)
    # MSE_mcts_refine = compute_mse(y_gt, y_mcts_refine)

    y_mcts_refine_train, y_mcts_refine_test , _, _ = refine_for_sample_test(model, equation_env, s,  samples['x_to_fit'], samples['y_to_fit'],samples['x_to_pred'])
    MSE_mcts_refine = compute_mse(y_gt_test, y_mcts_refine_test)
    

    print('#'*40)
    print('\nPre-trained E2E NMSE:', NMSE_ref)
    print('Pre-trained E2E NMSE after Refine:', NMSE_ref_refine)
    print('Pre-trained E2E NMSE (Beam Search):', NMSE_ref_search)
    print('Pre-trained E2E NMSE (Sampling):', NMSE_ref_sample)
    print('Pre-trained E2E MSE after Refine:', MSE_ref_refine)
    print('#'*20)
    print('TPSR+E2E NMSE:', NMSE_mcts)
    print('TPSR+E2E NMSE after Refine', NMSE_mcts_refine)
    print('TPSR+E2E MSE after Refine', MSE_mcts_refine)
    print('TPSR+E2E Time Elapsed:', time_elapsed)
    print('TPSR+E2E Sample Times (# of Explored Equation Candidates):', dp.sample_times)
    print('#'*40)

    replace_ops = {"add": "+", "mul": "*", "sub": "-", "pow": "**", "inv": "1/"}
    for op,replace_op in replace_ops.items():
        ref_str = ref_str.replace(op,replace_op)
        ref_str_sample = ref_str_sample.replace(op,replace_op)
        ref_str_search = ref_str_search.replace(op,replace_op)
        mcts_str = mcts_str.replace(op,replace_op)

    mcts_eq = sp.parse_expr(mcts_str)
    mcts_eq_refine = sp.parse_expr(mcts_str_refine)
    ref_eq = sp.parse_expr(ref_str)
    ref_eq_search = sp.parse_expr(ref_str_search)
    ref_eq_sample = sp.parse_expr(ref_str_sample)
    ref_eq_refine = sp.parse_expr(ref_str_refine)

    print("Pre-trained E2E Equation:", ref_eq)
    print("\nPre-trained E2E Equation after Refine: ", ref_eq_refine)
    print("\nPre-trained E2E Equation (Beam Search): ", ref_eq_search)
    print("\nPre-trained E2E Equation (Sampling): ", ref_eq_sample)
    print('#'*20)
    print("\nTPSR+E2E Equation: ", mcts_eq)
    print("\nTPSR+E2E Equation after Refine: ", mcts_eq_refine)
    print('#'*40)



def main_nesymres(case,params,eq_setting,cfg,samples,X,y):
    ## Set up BFGS load rom the hydra config yaml
    bfgs = BFGSParams(
            activated= cfg.inference.bfgs.activated,
            n_restarts=cfg.inference.bfgs.n_restarts,
            add_coefficients_if_not_existing=cfg.inference.bfgs.add_coefficients_if_not_existing,
            normalization_o=cfg.inference.bfgs.normalization_o,
            idx_remove=cfg.inference.bfgs.idx_remove,
            normalization_type=cfg.inference.bfgs.normalization_type,
            stop_time=cfg.inference.bfgs.stop_time,
        )

    params_fit = FitParams(word2id=eq_setting["word2id"], 
                                id2word={int(k): v for k,v in eq_setting["id2word"].items()}, 
                                una_ops=eq_setting["una_ops"], 
                                bin_ops=eq_setting["bin_ops"], 
                                total_variables=list(eq_setting["total_variables"]),  
                                total_coefficients=list(eq_setting["total_coefficients"]),
                                rewrite_functions=list(eq_setting["rewrite_functions"]),
                                bfgs=bfgs,
                                beam_size=cfg.inference.beam_size #This parameter is a tradeoff between accuracy and fitting time
                                )

    weights_path = "./nesymres/weights/10M.ckpt"

    ## Load architecture, set into eval mode, and pass the config parameters
    model = Model.load_from_checkpoint(weights_path, cfg=cfg.architecture)
    model.eval()
    if torch.cuda.is_available(): 
        model.cuda()

    fitfunc = partial(model.fitfunc,cfg_params=params_fit)

    output_ref = fitfunc(X,y) 

    ### MCTS 
    rl_env = RLEnv(
        params = params,
        samples = samples,
        model = model,
        cfg_params=params_fit)

    ## Get self.encoded in the model to use for Sequence generation from given states
    model.to_encode(X,y, params_fit)

    dp = NesymresHeuristic(
            rl_env=rl_env,
            model=model,
            k=params.width,
            num_beams=params.num_beams,
            horizon=params.horizon,
            device=params.device,
            use_seq_cache=not params.no_seq_cache,
            use_prefix_cache=not params.no_prefix_cache,
            length_penalty = params.beam_length_penalty,
            cfg_params = params_fit,
            train_value_mode=params.train_value,
            debug=params.debug)

    # for fair comparison, loading models and tokenizers are not included in computation time
    start = time.time()

    agent = UCT(
        action_space=[],
        gamma=1., 
        ucb_constant=1.,
        horizon=params.horizon,
        rollouts=params.rollout,
        dp=dp,
        width=params.width,
        reuse_tree=True
    )

    agent.display()

    if params.sample_only:
        horizon = 1
    else:
        horizon = 200

    done = False
    s = rl_env.state
    for t in range(horizon):
        if len(s) >= params.horizon:
            print(f'Cannot process programs longer than {params.horizon}. Stop here.')
            break

        if done:
            break

        act = agent.act(rl_env, done)
        s, r, done, _ = rl_env.step(act)

        if params.debug:
            # print the current tree
            print('tree:')
            print_tree(agent.root, params_fit.id2word)

            print('took action:')
            print(repr(params_fit.id2word[act]))
            print('========== state (excluding prompt) ==========')
            print(s)

        update_root(agent, act, s)
        dp.update_cache(s)

    time_elapsed = time.time() - start
    print("NeSymReS Equation Skeleton: ", output_ref)
    print("time elapsed: ", time_elapsed)
    print("samples times: ", dp.sample_times)
    print("generated ids: ", s)

    loss_bfgs_mcts , reward_mcts , pred_str = compute_reward_nesymres(model.X, model.y, s, params_fit)

    print("TPSR+NeSymReS Equation: ", pred_str)
    print("TPSR+NeSymReS Loss: ", loss_bfgs_mcts)
    print("TPSR+NeSymReS Reward: ", reward_mcts)




if __name__ == '__main__':

    case = 1
    parser = get_parser()
    params = parser.parse_args()
    np.random.seed(params.seed)
    torch.manual_seed(params.seed)
    torch.cuda.manual_seed(params.seed)
    params.debug = True
    params.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    
    if params.backbone_model == 'e2e':
        equation_env = build_env(params)
        modules = build_modules(equation_env, params)
        if not params.cpu:
            assert torch.cuda.is_available()
        symbolicregression.utils.CUDA = not params.cpu
        trainer = Trainer(modules, equation_env, params)
        
        #Example of Equation-Data:
        # x0 = np.random.uniform(-2,2, 200)
        x0 = np.linspace(-2,2, 200)
        # y = (x0 **2) * np.sin(x0)
        y= (x0**2 ) * np.sin(5*x0) + np.exp(-0.5*x0)
        data = np.concatenate((x0.reshape(-1,1),y.reshape(-1,1)), axis=1)
        samples = {'x_to_fit': 0, 'y_to_fit':0,'x_to_pred':0,'y_to_pred':0}
        samples['x_to_fit'] = [data[:,:1]]
        samples['y_to_fit'] = [data[:,1].reshape(-1,1)]
        samples['x_to_pred'] = [data[:,:1]]
        samples['y_to_pred'] = [data[:,1].reshape(-1,1)]
        
        #Main
        main_e2e(case, params, equation_env, samples) 
   
   
            
    if params.backbone_model == 'nesymres':
        with open('nesymres/jupyter/100M/eq_setting.json', 'r') as json_file:
            eq_setting = json.load(json_file)
        cfg = omegaconf.OmegaConf.load("nesymres/jupyter/100M/config.yaml")
        
        #Example of Equation-Data:
        number_of_points = 500
        n_variables = 2
        max_supp = cfg.dataset_train.fun_support["max"] 
        min_supp = cfg.dataset_train.fun_support["min"]
        X = torch.rand(number_of_points,len(list(eq_setting["total_variables"])))*(max_supp-min_supp)+min_supp
        X[:,n_variables:] = 0
        target_eq = "((x_1+0.76)*sin(0.8*exp(x_2))+(0.5*x_2))" #Use x_1,x_2 and x_3 as independent variables
        # target_eq = "((x_1*sin(x_2)+x_3))" #Use x_1,x_2 and x_3 as independent variables
        X_dict = {x:X[:,idx].cpu() for idx, x in enumerate(eq_setting["total_variables"])} 
        y = lambdify(",".join(eq_setting["total_variables"]), target_eq)(**X_dict)
        samples = {'x_to_fit':0, 'y_to_fit':0}
        samples['x_to_fit'] = [X]
        samples['y_to_fit'] = [y]
        
        #Main
        main_nesymres(case,params,eq_setting,cfg,samples,X,y)
    
