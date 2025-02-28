"""
UCT Algorithm

Required features of the environment:
env.state
env.action_space
env.transition(s ,a , is_model_dynamic)
env.equality_operator(s1, s2)
"""

import itertools
import dyna_gym.agents.mcts as mcts
import dyna_gym.utils.utils as utils
from math import sqrt, log
from gym import spaces

def uct_tree_policy(ag, children):
    return max(children, key=ag.ucb)

def p_uct_tree_policy(ag, children):
    return max(children, key=ag.p_ucb)

def var_p_uct_tree_policy(ag, children):
    return max(children, key=ag.var_p_ucb)


class UCT(object):
    """
    UCT agent
    """
    def __init__(self, action_space, rollouts=100, horizon=100, gamma=0.9, ucb_constant=6.36396103068, is_model_dynamic=True,
                 width=None, dp=None, reuse_tree=False,alg='var_p_uct',ucb_base=50):
        if type(action_space) == spaces.discrete.Discrete:
            self.action_space = list(mcts.combinations(action_space))
        else:
            self.action_space = action_space
            
        self.n_actions = len(self.action_space)
        self.rollouts = rollouts
        self.horizon = horizon
        self.gamma = gamma
        self.ucb_constant = ucb_constant
        self.is_model_dynamic = is_model_dynamic
        self.width = width or self.n_actions
        self.dp = dp
        self.reuse_tree = reuse_tree
        
        if alg == 'uct':
            self.tree_policy = uct_tree_policy
        elif alg == 'p_uct':
            self.tree_policy = p_uct_tree_policy
        elif alg == 'var_p_uct':
            self.tree_policy = var_p_uct_tree_policy
            self.ucb_base = ucb_base
            
        self.root = None


    def reset(self, p=None):
        """
        Reset the attributes.
        Expect to receive them in the same order as init.
        p : list of parameters
        """
        if p == None:
            self.__init__(self.action_space)
        else:
            utils.assert_types(p,[spaces.discrete.Discrete, int, int, float, float, bool])
            self.__init__(p[0], p[1], p[2], p[3], p[4], p[5])


    def display(self):
        """
        Display infos about the attributes.
        """
        print('Displaying UCT agent:')
        print('Number of actions  :', self.n_actions)
        print('Rollouts           :', self.rollouts)
        print('Horizon            :', self.horizon)
        print('Gamma              :', self.gamma)
        print('UCB constant       :', self.ucb_constant)
        print('Is model dynamic   :', self.is_model_dynamic)
        print('Expansion Width    :', self.width)
        print()


    def ucb(self, node):
        """
        Upper Confidence Bound of a chance node
        """
        return mcts.chance_node_value(node) + self.ucb_constant * sqrt(log(node.parent.visits)/len(node.sampled_returns))
    
    
    def p_ucb(self, node):
        """
        Upper Confidence Bound of a chance node, weighted by prior probability
        """
        return mcts.chance_node_value(node)\
            + self.ucb_constant * node.prob * sqrt(log(node.parent.visits)) / (len(node.sampled_returns))
            
    
    def var_p_ucb(self, node):
        """
        Upper Confidence Bound of a chance node, the ucb exploration weight is a variable
        """
        ucb_parameter = log((node.parent.visits + self.ucb_base + 1) / self.ucb_base) + self.ucb_constant
        return mcts.chance_node_value(node)\
            + ucb_parameter * node.prob * sqrt(log(node.parent.visits)) / (len(node.sampled_returns)) 


    def act(self, env, done):
        root = self.root if self.reuse_tree else None
        opt_act, self.root = mcts.mcts_procedure(self, self.tree_policy, env, done, root=root)
        return opt_act