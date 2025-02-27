"""
OLUCT (Open Loop UCT) Algorithm
"""

import gym
import random
import itertools
import dyna_gym.utils.utils as utils
from math import sqrt, log
from copy import copy

def value(node):
    """
    Value of the node
    """
    return sum(node.sampled_returns) / len(node.sampled_returns)

def combinations(space):
    if isinstance(space, gym.spaces.Discrete):
        return range(space.n)
    elif isinstance(space, gym.spaces.Tuple):
        return itertools.product(*[combinations(s) for s in space.spaces])
    else:
        raise NotImplementedError

class Node:
    def __init__(self, parent, action):
        self.parent = parent
        self.action = action
        if self.parent is None: # Root node
            self.depth = 0
        else: # Non root node
            self.depth = parent.depth + 1
        self.children = []
        self.explored_children = 0
        self.sampled_returns = []

class OLUCT(object):
    """
    OLUCT agent
    """
    def __init__(self, gamma=0.9, rollouts=100, max_depth=300, ucb_constant=6.36396103068, is_model_dynamic=True):
        self.gamma = gamma
        self.rollouts = rollouts
        self.max_depth = max_depth
        self.ucb_constant = ucb_constant
        self.is_model_dynamic = is_model_dynamic

    def reset(self, p=None):
        """
        Reset the attributes.
        Expect to receive them in the same order as init.
        p : list of parameters
        """
        if p == None:
            self.__init__(self.action_space)
        else:
            utils.assert_types(p,[float, int, int, float, bool])
            self.__init__(p[0], p[1], p[2], p[3], p[4])

    def display(self):
        """
        Display infos about the attributes.
        """
        print('Displaying OLUCT agent:')
        print('Gamma              :', self.gamma)
        print('Rollouts           :', self.rollouts)
        print('Max depth          :', self.max_depth)
        print('UCB constant       :', self.ucb_constant)
        print('Is model dynamic   :', self.is_model_dynamic)

    def ucb(self, node):
        """
        Upper Confidence Bound
        """
        return value(node) + self.ucb_constant * sqrt(log(len(node.parent.sampled_returns))/len(node.sampled_returns))

    def act(self, env, done):
        """
        Compute the entire OLUCT procedure
        """
        root = Node(None, None)
        for _ in range(self.rollouts):
            rewards = []
            node = root
            terminal = done
            state = env.state

            # Selection
            while len(node.children) != 0: 
                if node.explored_children < len(node.children):
                    child = node.children[node.explored_children]
                    node.explored_children += 1
                    node = child
                else:
                    node = max(node.children, key=self.ucb)
                state, reward, terminal = env.transition(state,node.action,self.is_model_dynamic)
                rewards.append(reward)

            # Expansion
            if not terminal:
                node.children = [Node(node, a) for a in combinations(env.action_space)]
                random.shuffle(node.children)

            # Evaluation
            t = 0
            estimate = 0
            while not terminal:
                action = env.action_space.sample()
                state, reward, terminal = env.transition(state,action,self.is_model_dynamic)
                estimate += reward * (self.gamma**t)
                t += 1
                if node.depth + t > self.max_depth:
                    break

            # Backpropagation
            while node:
                node.sampled_returns.append(estimate)
                if len(rewards) != 0:
                    estimate = rewards.pop() + self.gamma * estimate
                node = node.parent
        return max(root.children, key=value).action
