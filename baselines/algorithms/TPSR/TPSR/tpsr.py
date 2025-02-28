import json
import time
from symbolicregression.e2e_model import Transformer

from dyna_gym.agents.uct import UCT
from dyna_gym.agents.mcts import update_root, convert_to_json
from rl_env import RLEnv
from default_pi import E2EHeuristic


def tpsr_fit(model, scaled_X, Y, params, equation_env,bag_number=1,rescale=True):

    x_to_fit = scaled_X[0][(bag_number-1)*params.max_input_points:bag_number*params.max_input_points]
    y_to_fit = Y[0][(bag_number-1)*params.max_input_points:bag_number*params.max_input_points]

    samples = {'x_to_fit': 0, 'y_to_fit':0,'x_to_pred':0,'y_to_pred':0}
    samples['x_to_fit'] = [x_to_fit]
    samples['y_to_fit'] = [y_to_fit]
    model = Transformer(model=model,params = params, env=equation_env, samples=samples)
    model.to(params.device) 
    

    rl_env = RLEnv(
        params=params,
        samples = samples,
        equation_env = equation_env,
        model = model
    )

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
            debug=params.debug
        )

    # for fair comparison, loading models and tokenizers are not included in computation time
    start = time.time()

    agent = UCT(
        action_space=[],
        gamma=1., 
        ucb_constant=params.ucb_constant,
        horizon=params.horizon,
        rollouts=params.rollout,
        dp=dp,
        width=params.width,
        reuse_tree=True,
        alg=params.uct_alg,
        ucb_base=params.ucb_base
    )

    # agent.display()

    if params.sample_only:
        horizon = 1
    else:
        horizon = 200
        
    # try:
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

        if params.debug:
            print('tree:')
            # print_tree(agent.root, equation_env.equation_id2word)
            ret = convert_to_json(agent.root, rl_env, equation_env.equation_id2word[act])
            ret_all.append(ret)
            
            with open("tree_sample1.json", "w") as outfile:
                json.dump(ret_all, outfile,indent=1)

            print('took action:')
            print(repr(equation_env.equation_id2word[act]))
            print('========== state (excluding prompt) ==========')
            print(s)

        update_root(agent, act, s)
        dp.update_cache(s)
            
    time_elapsed = time.time() - start
    
    return s , time_elapsed, dp.sample_times
    
    