from typing import Callable, List, Optional, Union


class Expression:
    def __init__(self, child_num: int, func: Union[Callable, float], str_func: Union[Callable, str],
                 str_name: Optional[str] = None):
        """
        :param child_num: number of arguments e.g. add/mul has two arguments sin/cos only one
        :param func: numpy computation expression
        :param str_func: character computation expression
        """
        self.child = child_num
        self.func = func
        self.type = self.type_name = None
        self.str_func = str_func
        self.str_name = str_name


class TreeNode:
    """
    Nodes of the expression tree
    """

    def __init__(self, expr: Expression):
        """
        :param expr: the content of that expression
        """
        self._expr: Expression = expr
        self._children: List[TreeNode] = []
        self._last_num: int = expr.child
        self.depth: int = 1
        self.parent: Optional[TreeNode] = None
        self.tri_count: int = 1 if expr.type_name in ("Cos", "Sin") else 0

    def add_child(self, child) -> None:
        """
        Add a child node and set the node as the parent of the new node.
        """
        self._children.append(child)
        child.parent = self
        self._last_num -= 1
        child.depth = self.depth + 1
        child.tri_count += self.tri_count

    def is_full(self) -> bool:
        """
        :return: True if the subtree starts with this node full, False otherwise
        """
        return not self._last_num

    def traverse(self) -> List[int]:
        """
        :return: Sequence of tokens computed recursively
        """
        ans = [self._expr.type]
        for child in self._children:
            ans.extend(child.traverse())
        return ans

    def exp_str(self) -> str:
        """
        :return: Symbolic expressions for recursive computation
        """
        if self._expr.child:
            return self._expr.str_func(*[child.exp_str() for child in self._children])
        return self._expr.str_func

    def traverse_with_list(self, lst) -> List[List[int]]:
        """
        :return: Recursively compute the set of symbolic expressions for each subtree
        """
        ans = [self._expr.type]
        for child in self._children:
            ans.extend(child.traverse_with_list(ans))
        if 1 < len(ans) <= 10:
            lst.append(ans)
        return ans

    def func_name(self) -> str:
        """
        :return: the name of the function of this node
        """
        return self._expr.type_name
