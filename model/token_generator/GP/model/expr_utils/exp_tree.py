from typing import List, Optional
import abc

from .exp_tree_node import TreeNode


class TreeBase(abc.ABC):
    """
    Abstract base of expression trees
    """

    def __init__(self):
        super().__init__()
        self.root: Optional[TreeNode] = None
        self.node_stack: List[TreeNode] = []
        self.const_num: int = 0
        self.token_num: int = 0
        self.tri_tol: int = 0
        self.max_depth: int = 0

    @property
    def token_list_pre(self) -> list:
        """
        :return:Expression Tree Priority Traversal Results
        """
        if not self.root:
            return []
        return self.root.traverse()

    def depth(self):
        """
        :return: Node depth of the current insertion position node
        """
        if not self.node_stack:
            return 0
        return self.node_stack[self.top()].depth

    def trim(self):
        """
        Keep taking elements out of the heap until the element at the top of the heap is not filled.
        """
        while self.node_stack and self.node_stack[self.top()].is_full():
            self.max_depth = max(self.max_depth, self.node_stack[self.top()].depth)
            self.node_stack.pop(self.top())

    def is_full(self):
        """
        :return: True if the whole tree is full, False otherwise
        """
        self.trim()
        return self.root and not self.node_stack

    def get_exp(self):
        """
        :return: expression converted from expression tree
        """
        if not self.root:
            return ""
        return self.root.exp_str()

    @abc.abstractmethod
    def top(self) -> int:
        """
        :return: index of the top of the tree
        """
        pass

    def add_exp(self, exp):
        """
        add an expression to the 'top' node
        :param exp: new node
        """
        if self.is_full():
            raise RuntimeError(f"{self.token_list_pre} {exp.type_name}")
        self.trim()
        self.token_num += 1
        if exp.type_name == "C":
            self.const_num += 1
        node = TreeNode(exp)
        if not self.root:
            self.root = node
            self.node_stack.append(node)
            return
        self.tri_tol += node.tri_count
        self.node_stack[self.top()].add_child(node)
        self.node_stack.append(node)

    @property
    def head_token(self):
        """
        :return: function name of the head node
        """
        self.trim()
        if not self.root:
            return ""
        return self.node_stack[self.top()].func_name()

    def pre_lists(self):
        """
        :return: Sequences of tokens computed recursively of each node in Expression tree
        """
        lst = []
        self.root.traverse_with_list(lst=lst)
        return lst

    @property
    def tri_count(self):
        """
        :return: the number of triangles tokens (sin/cos/...) in the fathers of  current insertion position node
        """
        self.trim()
        if not self.node_stack:
            return 0
        return self.node_stack[self.top()].tri_count



class PreTree(TreeBase):
    """
    Expression tree with elements added in pre-order for genetic algorithms
    """

    def top(self) -> int:
        return -1
