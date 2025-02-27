"""
Asynchronous Dynamic Programming using tree structure

Required attributes and functions of the environment:
env.get_time()
env.is_terminal(state)
env.static_reachable_states(s, a)
env.dynamic_reachable_states(s, a)
env.transition_probability(s_p, s, t, a)
env.instant_reward(s, t, a, s_p)
env.expected_reward(s, t, a)
"""

import random
import numpy as np
import dyna_gym.utils.utils as utils
from gym import spaces

def node_value(node):
    assert node.value != None, 'Error: node value={}'.format(node.value)
    return node.value

class DecisionNode:
    """
    Decision node class, labelled by a state
    """
    def __init__(self, parent, state, weight, is_terminal):
        self.parent = parent
        self.state = state
        self.weight = weight # Probability to occur
        self.is_terminal = is_terminal
        if self.parent is None: # Root node
            self.depth = 0
        else: # Non root node
            self.depth = parent.depth + 1
        self.children = []
        self.value = None

class ChanceNode:
    """
    Chance node class, labelled by a state-action pair
    The state is accessed via the parent attribute
    """
    def __init__(self, parent, action):
        self.parent = parent
        self.action = action
        self.depth = parent.depth
        self.children = []
        self.value = None

class AsynDP(object):
    """
    AsynDP agent
    """
    def __init__(self, action_space, gamma=0.9, max_depth=4, is_model_dynamic=True):
        self.action_space = action_space
        self.n_actions = self.action_space.n
        self.gamma = gamma
        self.max_depth = max_depth
        self.is_model_dynamic = is_model_dynamic
        self.t_call = None

    def reset(self, p=None):
        """
        Reset the attributes.
        Expect to receive them in the same order as init.
        p : list of parameters
        """
        if p == None:
            self.__init__(self.action_space)
        else:
            utils.assert_types(p,[spaces.discrete.Discrete, float, int, bool])
            self.__init__(p[0], p[1], p[2], p[3])

    def display(self):
        """
        Display infos about the attributes
        """
        print('Displaying AsynDP agent:')
        print('Action space      :', self.action_space)
        print('Number of actions :', self.n_actions)
        print('Gamma             :', self.gamma)
        print('Maximum depth     :', self.max_depth)
        print('Is model dynamic  :', self.is_model_dynamic)

    def build_tree(self, node, env):
        if type(node) is DecisionNode: #DecisionNode
            if (node.depth < self.max_depth):
                for a in range(self.n_actions):
                    node.children.append(ChanceNode(node, a))
            else: #Reached maximum depth
                return None
        else: #ChanceNode
            if self.is_model_dynamic:
                for s_p in env.dynamic_reachable_states(node.parent.state, node.action):
                    node.children.append(
                        DecisionNode(
                            parent=node,
                            state=s_p,
                            weight=env.transition_probability(s_p, node.parent.state, node.parent.state.time, node.action),
                            is_terminal=env.is_terminal(s_p)
                        )
                    )
            else:
                for s_p in env.static_reachable_states(node.parent.state, node.action):
                    node.children.append(
                        DecisionNode(
                            parent=node,
                            state=s_p,
                            weight=env.transition_probability(s_p, node.parent.state, self.t_call, node.action),
                            is_terminal=env.is_terminal(s_p)
                        )
                    )
        for ch in node.children:
            if type(ch) is DecisionNode:
                if not ch.is_terminal:
                    self.build_tree(ch, env)
            else: #ChanceNode
                self.build_tree(ch, env)

    def initialize_tree(self, env, done):
        """
        Initialize an empty tree.

        The tree is composed with all the possible actions as chance nodes and all the
        possible state as decision nodes.

        The used model is either static (i.e. snapshot) or dynamic w.r.t. the
        is_model_dynamic variable.

        The depth of the tree is defined by the self.max_depth attribute of the agent.

        The used heuristic for the evaluation of the leaf nodes that are not terminal
        nodes is defined by the function self.heuristic_value.
        """
        root = DecisionNode(None, env.state, 1, done)
        self.build_tree(root, env)
        return root

    def fill_tree(self, node, env):
        """
        Fill the tree with Bellman equation updates and bootstraps with heuristic function.
        """
        if (type(node) is DecisionNode):
            if (node.depth == self.max_depth):
                assert node.value == None, 'Error: node value={}'.format(node.value)
                node.value = self.heuristic_value(node, env)
            elif node.is_terminal:
                assert node.value == None, 'Error: node value={}'.format(node.value)
                if self.is_model_dynamic:
                    node.value = env.instant_reward(node.parent.parent.state, node.parent.parent.state.time, node.parent.action, node.state)
                else:
                    node.value = env.instant_reward(node.parent.parent.state, self.t_call, node.parent.action, node.state)
            else:
                v = -1e99
                for ch in node.children:
                    v = max(v, self.fill_tree(ch, env))
                assert node.value == None, 'Error: node value={}'.format(node.value)
                node.value = v
        else: # ChanceNode
            v = 0.0
            for ch in node.children:
                v += ch.weight * self.fill_tree(ch, env)
            v *= self.gamma
            if self.is_model_dynamic:
                v += env.expected_reward(node.parent.state, node.parent.state.time, node.action)
            else:
                v += env.expected_reward(node.parent.state, self.t_call, node.action)
            assert node.value == None, 'Error: node value={}'.format(node.value)
            node.value = v
        return node.value

    def heuristic_value(self, node, env):
        return 0.0

    def test(self, v):
        print('s0 :', v.state.index, 'value :', v.value, 'terminal :', v.is_terminal)
        for c in v.children:
            print('-> a', c.action, 'value :', c.value)
            for cc in c.children:
                print('      s1 :', cc.state.index, 'weight :', cc.weight, 'value :', cc.value, 'terminal :', cc.is_terminal)
                for ccc in cc.children:
                    print('         a', ccc.action, 'value :', ccc.value)
                    for cccc in ccc.children:
                        print('            s2 :', cccc.state.index, 'weight :', cccc.weight, 'value :', cccc.value, 'terminal :', cccc.is_terminal)

    def act(self, env, done):
        """
        Compute the entire AsynDP procedure
        """
        self.t_call = env.get_time()
        root = self.initialize_tree(env, done)
        self.fill_tree(root, env)
        return max(root.children, key=node_value).action
