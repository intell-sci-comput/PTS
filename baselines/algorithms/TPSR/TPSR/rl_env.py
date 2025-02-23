import logging
import os
from collections import OrderedDict
from types import SimpleNamespace
import time 

from reward import compute_reward_e2e, compute_reward_nesymres
from symbolicregression.e2e_model import refine_for_sample

class RLEnv:
    """
    Equation Generation RL environment.

    State: a list of tokens.
    Action: a token (an integer).
    Reward: Fittness reward of the generated equation.
    """
    def __init__(self,samples, params=None, equation_env=None, model=None, cfg_params=None):
        self.params = params
        self.samples = samples
        self.equation_env = equation_env
        self.model = model
        self.cfg_params = cfg_params

        if self.params.backbone_model == 'e2e':
            self.state = [self.equation_env.equation_word2id['<EOS>']]
            self.terminal_token = self.equation_env.equation_word2id['<EOS>']
            
        elif self.params.backbone_model == 'nesymres':
            self.state = [cfg_params.word2id["S"]]
            self.terminal_token = cfg_params.word2id["F"]
        # state -> reward
        # we may need to retrieve the states (programs) in the order they were saved, so use OrderedDict
        self.cached_reward = OrderedDict()


    def transition(self, s, a, is_model_dynamic=True):
        if a == self.terminal_token:
            done = True
        else:
            done = False
        next_state = s + [a]
        if done:
            reward = self.get_reward(next_state)
        else:
            reward = 0 # no intermediate reward
        
        return next_state, reward, done


    def step(self, action):
        self.state, reward, done = self.transition(self.state, action)

        return self.state, reward, done, {}


    def get_reward(self, s,mode='train'):
        """
        Returns:
            The reward of program in s.
        """
        if s is None:
            return 0

        if tuple(s) in self.cached_reward.keys() and mode == 'train':
            # cache rewards for training
            return self.cached_reward[tuple(s)]

        if self.params.backbone_model == 'e2e':
            if (type(s) != list):
                s = s.tolist()
            y_pred, model_str, generations_tree = refine_for_sample(self.params, self.model,self.equation_env, s, x_to_fit = self.samples['x_to_fit'],y_to_fit = self.samples['y_to_fit']) 
            reward = compute_reward_e2e(self.params,self.samples, y_pred, model_str, generations_tree)

        if self.params.backbone_model == 'nesymres':
            start_time = time.time()
            _, reward, _ = compute_reward_nesymres(self.model.X ,self.model.y, s, self.cfg_params)
            print("time to get reward: ", time.time() - start_time) #bfgs for nesymres is time-consuming
         
        if mode == 'train':
            self.cached_reward[tuple(s)] = reward

        return reward

    def equality_operator(self, s1, s2):
        return s1 == s2
    
    def tokenizer_decode(self, node_action):
        return self.equation_env.equation_id2word[node_action]

    def convert_state_to_program(self, state):
        prog = []
        if type(state) != list:
            state = state.tolist()
        for i in range(len(state)):
            prog.append(self.equation_env.equation_id2word[state[i]])
        # return prog
        return " ".join(prog)