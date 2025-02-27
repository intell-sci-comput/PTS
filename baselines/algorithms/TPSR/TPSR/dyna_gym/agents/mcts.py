"""
MCTS Algorithm

Required features of the environment:
env.state
env.action_space
env.transition(s ,a , is_model_dynamic)
env.equality_operator(s1, s2)
"""
import logging
import random
import itertools
import warnings
from tqdm import tqdm

import dyna_gym.utils.utils as utils
from gym import spaces
import numpy as np
from math import sqrt, log
from copy import copy

import networkx as nx
import matplotlib.pyplot as plt
from eval.utils import hierarchy_pos


def chance_node_value(node):
    """
    Value of a chance node
    """
    return max(node.sampled_returns)

def decision_node_value(node):
    """
    Value of a decision node
    """
    return chance_node_value(max(node.children, key=chance_node_value))

def combinations(space):
    if isinstance(space, spaces.Discrete):
        return range(space.n)
    elif isinstance(space, spaces.Tuple):
        return itertools.product(*[combinations(s) for s in space.spaces])
    else:
        raise NotImplementedError

def mcts_tree_policy(ag, children):
    return random.choice(children)

def mcts_procedure(ag, tree_policy, env, done, root=None):
    """
    Compute the entire MCTS procedure wrt to the selected tree policy.
    Funciton tree_policy is a function taking an agent + a list of ChanceNodes as argument
    and returning the one chosen by the tree policy.
    """
    decision_node_num = 0
    if root is not None:
        assert root.state == env.state
    else:
        root = DecisionNode(None, env.state, ag.action_space.copy(), done, dp=ag.dp,id=decision_node_num)
        decision_node_num += 1

    for _ in range(ag.rollouts):
        rewards = []
        node = root 
        terminal = done

        # Selection
        select = True
        while select:
            if (type(node) == DecisionNode): 
                if node.is_terminal:
                    select = False 
                else:
                    if len(node.possible_actions) > 0:
                        select = False 
                    else:
                        node = tree_policy(ag, node.children)
            else: # ChanceNode
                state_p, reward, terminal = env.transition(node.parent.state, node.action, ag.is_model_dynamic)
                rewards.append(reward)
                if (len(node.children) == 0):
                    select = False 
                else:
                    new_state = True
                    for i in range(len(node.children)):
                        if env.equality_operator(node.children[i].state, state_p):
                            node = node.children[i]
                            new_state = False
                            break
                    if new_state:
                        select = False 

        # Expansion
        if (type(node) == ChanceNode) or ((type(node) == DecisionNode) and not node.is_terminal):
            if (type(node) == DecisionNode):
                node.children.append(ChanceNode(node, node.possible_actions.pop()))
                node = node.children[-1]
                state_p, reward, terminal = env.transition(node.parent.state, node.action, ag.is_model_dynamic)
                if len(state_p) >= ag.horizon:
                    terminal = True
                rewards.append(reward)
            node.children.append(DecisionNode(node, state_p, ag.action_space.copy(), terminal, dp=ag.dp,id=decision_node_num))
            decision_node_num += 1
            node = node.children[-1]

        # Evaluation
        assert(type(node) == DecisionNode)
        state = node.state
        if ag.dp is None:
            t = 0
            estimate = 0
            while (not terminal) and (t < ag.horizon):
                action = env.action_space.sample()
                state, reward, terminal = env.transition(state, action, ag.is_model_dynamic)
                estimate += reward * (ag.gamma**t)
                t += 1
        else:
            if not node.is_terminal:
                if ag.dp.is_train_value_mode:
                    state, hidden_states = ag.dp.get_predict_sequence(state, ret_states=True)
                    estimate = env.get_reward(state)

                    ag.dp.train_value_func(hidden_states, estimate)
                else:
                    state = ag.dp.get_predict_sequence(state)
                    estimate = env.get_reward(state)
                    node.info['complete_program'] = state
            else:
                estimate = 0

        # Backpropagation
        node.visits += 1
        node = node.parent
        assert(type(node) == ChanceNode)
        while node:
            if len(rewards) != 0:
                estimate = rewards.pop() + ag.gamma * estimate
            node.sampled_returns.append(estimate)
            node.parent.visits += 1
            node = node.parent.parent

        assert len(rewards) == 0
    return max(root.children, key=chance_node_value).action, root

class DecisionNode:
    """
    Decision node class, labelled by a state

    Args:
        dp: default policy, used to prioritize and filter possible actions
    """
    def __init__(self, parent, state, possible_actions, is_terminal, dp=None, id=None):
        self.id = id
        self.parent = parent
        self.state = state
        self.is_terminal = is_terminal
        if self.parent is None:
            self.depth = 0
        else: 
            self.depth = parent.depth + 1
        self.children = []
        if dp is None:
            self._possible_actions = possible_actions
            random.shuffle(self._possible_actions)
        else:
            self._possible_actions = None
            self.dp = dp

        self.explored_children = 0
        self.visits = 0
        self.info = {}

    @property
    def possible_actions(self):
        if self._possible_actions is not None:
            return self._possible_actions
        else:
            assert self.dp is not None, 'self._possible_acitons are not set in init, and no dp is provided.'
            top_k_predict = self.dp.get_top_k_predict(self.state)
            self._possible_actions = list(reversed(top_k_predict))
            return self._possible_actions

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
        self.sampled_returns = []

class MCTS(object):
    """
    MCTS agent
    """
    def __init__(self, action_space, rollouts=100, horizon=100, gamma=0.9, is_model_dynamic=True):
        if type(action_space) == spaces.discrete.Discrete:
            self.action_space = list(combinations(action_space))
        else:
            self.action_space = action_space
        self.n_actions = len(self.action_space)
        self.rollouts = rollouts
        self.horizon = horizon
        self.gamma = gamma
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
            utils.assert_types(p,[spaces.discrete.Discrete, int, int, float, bool])
            self.__init__(p[0], p[1], p[2], p[3], p[4])


    def display(self):
        """
        Display infos about the attributes.
        """
        print('Displaying MCTS agent:')
        print('Action space       :', self.action_space)
        print('Number of actions  :', self.n_actions)
        print('Rollouts           :', self.rollouts)
        print('Horizon            :', self.horizon)
        print('Gamma              :', self.gamma)
        print('Is model dynamic   :', self.is_model_dynamic)

    def act(self, env, done):
        opt_act, _, = mcts_procedure(self, mcts_tree_policy, env, done)
        return opt_act


def update_root(ag, act, state_p):
    root_updated = False
    for chance_node in ag.root.children:
        if act == chance_node.action:
            for decision_node in chance_node.children:
                if decision_node.state == state_p:
                    ag.root = decision_node
                    root_updated = True
                    break

    if not root_updated:
        raise Exception("root update fails, can't find the next state, action pair in tree.")


def pre_order_traverse(
        decision_node: DecisionNode,
        decision_node_fn=lambda n, d: None,
        chance_node_fn=lambda n, d: None,
        depth=0):
    """
    Postorder traversal of the tree rooted at state
    Apply fn once visited
    """
    decision_node_fn(decision_node, depth)

    for chance_node in decision_node.children:
        chance_node_fn(chance_node, depth)
        for next_decision_node in chance_node.children:
            pre_order_traverse(next_decision_node, decision_node_fn, chance_node_fn, depth + 1)

def print_tree(root: DecisionNode, equation_id2word):
    def printer(node: ChanceNode, depth):
        print("\t" * depth, repr(equation_id2word[node.action]), chance_node_value(node), node.sampled_returns)
    pre_order_traverse(root, chance_node_fn=printer, depth=0)
    
   
    
def plot_tree(root: DecisionNode, filename, equation_id2word):
    """
    Plot the tree rooted at root
    """

    def printer(node: ChanceNode, depth):
        print("\t" * depth, repr(equation_id2word[node.action]), chance_node_value(node), node.sampled_returns)

    pre_order_traverse(root, chance_node_fn=printer)

    G = nx.DiGraph()
    G.add_node(root.id, label='<PD>')

    def add_node(node: ChanceNode, depth):
        if len(node.children) > 0:
            child_id = node.children[0].id
            parent_id = node.parent.id

            G.add_node(child_id)
            G.add_edge(parent_id, child_id, label=repr(equation_id2word[node.action]))
    
    pre_order_traverse(root, chance_node_fn=add_node)

    plt.figure(figsize=(15, 15))

    pos = hierarchy_pos(G, root=root.id)
    nx.draw(G, pos, with_labels=True)

    edge_labels = nx.get_edge_attributes(G, 'label')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, label_pos=0.5, rotate=False)

    plt.savefig(filename + '.pdf', format="pdf")
    plt.close()




def convert_to_json(root: DecisionNode, env, selected_act):
    """
    Save the information of children of root into a list.
    Does not distinguish layers. So works when the tree only expands one level.
    """
    ret = []

    def get_info(node: ChanceNode, depth):
        if node.action == env.terminal_token:
            complete_program = env.convert_state_to_program(node.children[0].state)
        else:
            complete_program = env.convert_state_to_program(node.children[0].info['complete_program'])

        info = {'token': env.tokenizer_decode(node.action),
                'state': env.convert_state_to_program(node.children[0].state),
                'selected': node.action == selected_act,
                'score': chance_node_value(node),
                'complete_program': complete_program}
        ret.append(info)

    pre_order_traverse(root, chance_node_fn=get_info)
    
    return ret