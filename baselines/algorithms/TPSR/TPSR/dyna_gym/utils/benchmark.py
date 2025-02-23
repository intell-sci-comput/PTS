"""
Generic benchmark method:

Require:
agent.reset(param)
agent.display()
agent.act(env, done)
agent.gamma

env.reset()
"""

import gym
import csv
import numpy as np
import statistics as stat
import dyna_gym.agents.uct as uct
import dyna_gym.agents.my_random_agent as ra
import random
from multiprocessing import Pool

def csv_write(row, path, mode):
    with open(path, mode) as csvfile:
        w = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        w.writerow(row)

def run(agent, env, tmax, verbose=False):
    """
    Run single episode
    Return: (undiscounted_return, total_time, discounted_return)
    """
    done = False
    undiscounted_return, total_time, discounted_return = 0.0, 0, 0.0
    if verbose:
        env.render()
    for t in range(tmax):
        action = agent.act(env,done)
        _, r, done, _ = env.step(action)
        undiscounted_return += r
        discounted_return += (agent.gamma**t) * r
        if verbose:
            env.render()
        if (t+1 == tmax) or done:
            total_time = t+1
            break
    return undiscounted_return, total_time, discounted_return

def singlethread_benchmark(env_name, n_env, agent_name_pool, agent_pool, param_pool, param_names_pool, n_epi, tmax, save, paths_pool, verbose=True):
    """
    Benchmark multiple agents within a single environment.
    Single thread method.
    env_name         : name of the generated environment
    n_env            : number of generated environment
    agent_name_pool  : list containing the names of the agents for saving purpose
    agent_pool       : list containing the agent objects
    param_pool       : list containing lists of parameters for each agent object
    param_names_pool : list containing the parameters names
    n_epi            : number of episodes per generated environment
    tmax             : timeout for each episode
    save             : save the results or not
    paths_pool       : list containing the saving path for each agent
    verbose          : if true, display informations during benchmark
    """
    assert len(agent_name_pool) == len(agent_pool) == len(param_pool)
    n_agt = len(param_pool)
    if save:
        assert len(paths_pool) == n_agt
        for _agt in range(n_agt):
            csv_write(['env_name', 'env_number', 'agent_name', 'agent_number'] + param_names_pool[_agt] + ['epi_number', 'undiscounted_return', 'total_time', 'discounted_return'], paths_pool[_agt], 'w')
    for _env in range(n_env):
        env = gym.make(env_name)
        if verbose:
            print('Created environment', _env+1, '/', n_env)
        for _agt in range(n_agt):
            agt_name = agent_name_pool[_agt]
            agt = agent_pool[_agt]
            n_prm = len(param_pool[_agt])
            for _prm in range(n_prm):
                prm = param_pool[_agt][_prm]
                agt.reset(prm)
                if verbose:
                    print('Created agent', _agt+1, '/', n_agt,'with parameters', _prm+1, '/', n_prm)
                    agt.display()
                for _epi in range(n_epi):
                    if verbose:
                        print('Environment', env_name, _env+1, '/', n_env, 'agent', agt_name, _prm+1, '/', n_prm,'running episode', _epi+1, '/', n_epi)
                    env.reset()
                    undiscounted_return, total_time, discounted_return = run(agt, env, tmax)
                    if save:
                        csv_write([env_name, _env, agt_name, _prm] + prm + [_epi, undiscounted_return, total_time, discounted_return], paths_pool[_agt], 'a')

def multithread_run(env_name, _env, n_env, env, agt_name, _agt, n_agt, agt, _prm, n_prm, prm, tmax, n_epi, _thr, save, path, verbose, save_period):
    saving_pool = []
    for _epi in range(n_epi):
        if verbose:
            print('Environment', env_name, _env+1, '/', n_env, 'agent', agt_name, _prm+1, '/', n_prm,'running episode', _epi+1, '/', n_epi, '(thread nb', _thr, ')')
        env.reset()
        undiscounted_return, total_time, discounted_return = run(agt, env, tmax)
        if save:
            saving_pool.append([env_name, _env, agt_name, _prm] + prm + [_thr, undiscounted_return, total_time, discounted_return])
            if len(saving_pool) == save_period:
                for row in saving_pool:
                    csv_write(row, path, 'a')
                saving_pool = []
    if save:
        for row in saving_pool:
            csv_write(row, path, 'a')

def multithread_benchmark(env_name, n_env, agent_name_pool, agent_pool, param_pool, param_names_pool, n_epi, tmax, save, paths_pool, n_thread, verbose=True, save_period=1):
    """
    Benchmark multiple agents within a single environment.
    Multithread method.
    env_name         : name of the generated environment
    n_env            : number of generated environment
    agent_name_pool  : list containing the names of the agents for saving purpose
    agent_pool       : list containing the agent objects
    param_pool       : list containing lists of parameters for each agent object
    param_names_pool : list containing the parameters names
    n_epi            : number of episodes per generated environment
    tmax             : timeout for each episode
    save             : save the results or not
    paths_pool       : list containing the saving path for each agent
    n_thread         : number of threads
    verbose          : if true, display informations during benchmark
    """
    assert len(agent_name_pool) == len(agent_pool) == len(param_pool)
    pool = Pool(processes=n_thread)
    n_agt = len(param_pool)
    n_epi = int(n_epi / n_thread)
    if save:
        assert len(paths_pool) == n_agt
        for _agt in range(n_agt):
            csv_write(['env_name', 'env_number', 'agent_name', 'param_number'] + param_names_pool[_agt] + ['thread_number', 'undiscounted_return', 'total_time', 'discounted_return'], paths_pool[_agt], 'w')
    for _env in range(n_env):
        env = gym.make(env_name)
        if verbose:
            print('Created environment', _env+1, '/', n_env)
        for _agt in range(n_agt):
            agt_name = agent_name_pool[_agt]
            agt = agent_pool[_agt]
            n_prm = len(param_pool[_agt])
            for _prm in range(n_prm):
                prm = param_pool[_agt][_prm]
                agt.reset(prm)
                if verbose:
                    print('Created agent', _agt+1, '/', n_agt,'with parameters', _prm+1, '/', n_prm)
                    agt.display()
                results_pool = []
                for _thr in range(n_thread):
                    results_pool.append(pool.apply_async(multithread_run,[env_name, _env, n_env, env, agt_name, _agt, n_agt, agt, _prm, n_prm, prm, tmax, n_epi, _thr+1, save, paths_pool[_agt], verbose, save_period]))
                for result in results_pool:
                    result.get()

def test_multithread():
    env_name = 'NSFrozenLakeEnv-v0'
    n_env = 4
    n_epi = 32
    tmax = 100
    n_thread = 4

    env = gym.make(env_name)
    agent_name_pool = ['UCT', 'RANDOM']
    agent_pool = [uct.UCT(env.action_space), ra.MyRandomAgent(env.action_space)]
    param_names_pool = [
        ['action_space','rollouts','horizon','gamma','ucb_constant','is_model_dynamic'],
        ['action_space']
    ]
    param_pool = [
        [[env.action_space,  1, 1, 0.9, 6.36396103068, True],[env.action_space, 10, 10, 0.9, 6.36396103068, True]],
        [[env.action_space]]
    ]
    paths_pool = ['data/test_uct.csv','data/test_random.csv']

    multithread_benchmark(
        env_name         = env_name,
        n_env            = n_env,
        agent_name_pool  = agent_name_pool,
        agent_pool       = agent_pool,
        param_pool       = param_pool,
        param_names_pool = param_names_pool,
        n_epi            = n_epi,
        tmax             = tmax,
        save             = True,
        paths_pool       = paths_pool,
        n_thread         = n_thread,
        verbose          = True,
        save_period      = 1
    )

def test_singlethread():
    env_name = 'NSFrozenLakeEnv-v0'
    n_env = 2
    n_epi = 8
    tmax = 100

    env = gym.make(env_name)
    agent_name_pool = ['UCT','RANDOM']
    agent_pool = [uct.UCT(env.action_space), ra.MyRandomAgent(env.action_space)]
    param_names_pool = [
        ['action_space','rollouts','horizon','gamma','ucb_constant','is_model_dynamic'],
        ['action_space']
    ]
    param_pool = [
        [[env.action_space,  10, 100, 0.9, 6.36396103068, True],[env.action_space, 100, 100, 0.9, 6.36396103068, True]],
        [[env.action_space]]
    ]
    paths_pool = ['data/test_uct.csv','data/test_random.csv']

    singlethread_benchmark(
        env_name         = env_name,
        n_env            = n_env,
        agent_name_pool  = agent_name_pool,
        agent_pool       = agent_pool,
        param_pool       = param_pool,
        param_names_pool = param_names_pool,
        n_epi            = n_epi,
        tmax             = tmax,
        save             = True,
        paths_pool       = paths_pool,
        verbose          = True
    )
