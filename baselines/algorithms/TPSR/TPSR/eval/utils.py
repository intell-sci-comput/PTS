import pprint
import subprocess

import numpy as np

import networkx as nx
import random



def log_error(msg:str, log_loc:str):
    print(msg)
    with open(log_loc, 'w') as f:
        f.write(msg)


# output related
def colored_background(r, g, b, text):
    """
    Print text with (r, g, b) background
    """
    clip_and_int = lambda x: np.clip(int(x), 0, 256)
    return f'\033[48;2;{clip_and_int(r)};{clip_and_int(g)};{clip_and_int(b)}m{text}\033[0m'


def print_colored_output(char_and_uncertainties):
    """
    Args:
        char_and_uncertainties: a list of (char, uncertainty) pairs
    """
    uncertainties = [u for (c, u) in char_and_uncertainties]
    uncertainties = np.clip(uncertainties, 0, 1)
    hist, _ = np.histogram(uncertainties, bins=10, range=(0., 1.))
    print('uncertainty hist')
    print(hist)

    print('all char and uncertainties')
    pprint.pprint(char_and_uncertainties)

    for char, uncertainty in char_and_uncertainties:
        if '\n' in char:
            print(char, end='') # has issues in rendering \n, so ignore it
        else:
            r = int(np.clip(255 * uncertainty, 0, 255)) # clip to make sure it's in range, otherwise simply output black color, which would be hard to debug
            print(colored_background(r, 50, 50, char), end='')
    print()


def execute(command, working_dir='./'):
    print(f"Running {command}")
    return subprocess.Popen(command.split(' '), cwd=working_dir)


def hierarchy_pos(G, root=None, width=1., vert_gap=0.2, vert_loc=0, leaf_vs_root_factor=0.5):
    '''
    If the graph is a tree this will return the positions to plot this in a
    hierarchical layout.

    Based on Joel's answer at https://stackoverflow.com/a/29597209/2966723,
    but with some modifications.

    We include this because it may be useful for plotting transmission trees,
    and there is currently no networkx equivalent (though it may be coming soon).

    There are two basic approaches we think of to allocate the horizontal
    location of a node.

    - Top down: we allocate horizontal space to a node.  Then its ``k``
      descendants split up that horizontal space equally.  This tends to result
      in overlapping nodes when some have many descendants.
    - Bottom up: we allocate horizontal space to each leaf node.  A node at a
      higher level gets the entire space allocated to its descendant leaves.
      Based on this, leaf nodes at higher levels get the same space as leaf
      nodes very deep in the tree.

    We use use both of these approaches simultaneously with ``leaf_vs_root_factor``
    determining how much of the horizontal space is based on the bottom up
    or top down approaches.  ``0`` gives pure bottom up, while 1 gives pure top
    down.


    :Arguments:

    **G** the graph (must be a tree)

    **root** the root node of the tree
    - if the tree is directed and this is not given, the root will be found and used
    - if the tree is directed and this is given, then the positions will be
      just for the descendants of this node.
    - if the tree is undirected and not given, then a random choice will be used.

    **width** horizontal space allocated for this branch - avoids overlap with other branches

    **vert_gap** gap between levels of hierarchy

    **vert_loc** vertical location of root

    **leaf_vs_root_factor**

    xcenter: horizontal location of root
    '''
    if not nx.is_tree(G):
        raise TypeError('cannot use hierarchy_pos on a graph that is not a tree')

    if root is None:
        if isinstance(G, nx.DiGraph):
            root = next(iter(nx.topological_sort(G)))  # allows back compatibility with nx version 1.11
        else:
            root = random.choice(list(G.nodes))

    def _hierarchy_pos(G, root, leftmost, width, leafdx=0.2, vert_gap=0.2, vert_loc=0,
                       xcenter=0.5, rootpos=None,
                       leafpos=None, parent=None):
        '''
        see hierarchy_pos docstring for most arguments

        pos: a dict saying where all nodes go if they have been assigned
        parent: parent of this branch. - only affects it if non-directed

        '''

        if rootpos is None:
            rootpos = {root: (xcenter, vert_loc)}
        else:
            rootpos[root] = (xcenter, vert_loc)
        if leafpos is None:
            leafpos = {}
        children = list(G.neighbors(root))
        leaf_count = 0
        if not isinstance(G, nx.DiGraph) and parent is not None:
            children.remove(parent)
        if len(children) != 0:
            rootdx = width / len(children)
            nextx = xcenter - width / 2 - rootdx / 2
            for child in children:
                nextx += rootdx
                rootpos, leafpos, newleaves = _hierarchy_pos(G, child, leftmost + leaf_count * leafdx,
                                                             width=rootdx, leafdx=leafdx,
                                                             vert_gap=vert_gap, vert_loc=vert_loc - vert_gap,
                                                             xcenter=nextx, rootpos=rootpos, leafpos=leafpos,
                                                             parent=root)
                leaf_count += newleaves

            leftmostchild = min((x for x, y in [leafpos[child] for child in children]))
            rightmostchild = max((x for x, y in [leafpos[child] for child in children]))
            leafpos[root] = ((leftmostchild + rightmostchild) / 2, vert_loc)
        else:
            leaf_count = 1
            leafpos[root] = (leftmost, vert_loc)
        #        pos[root] = (leftmost + (leaf_count-1)*dx/2., vert_loc)
        #        print(leaf_count)
        return rootpos, leafpos, leaf_count

    xcenter = width / 2.
    if isinstance(G, nx.DiGraph):
        leafcount = len([node for node in nx.descendants(G, root) if G.out_degree(node) == 0])
    elif isinstance(G, nx.Graph):
        leafcount = len([node for node in nx.node_connected_component(G, root) if G.degree(node) == 1 and node != root])
    rootpos, leafpos, leaf_count = _hierarchy_pos(G, root, 0, width,
                                                  leafdx=width * 1. / leafcount,
                                                  vert_gap=vert_gap,
                                                  vert_loc=vert_loc,
                                                  xcenter=xcenter)
    pos = {}
    for node in rootpos:
        pos[node] = (
        leaf_vs_root_factor * leafpos[node][0] + (1 - leaf_vs_root_factor) * rootpos[node][0], leafpos[node][1])
    #    pos = {node:(leaf_vs_root_factor*x1+(1-leaf_vs_root_factor)*x2, y1) for ((x1,y1), (x2,y2)) in (leafpos[node], rootpos[node]) for node in rootpos}
    xmax = max(x for x, y in pos.values())
    for node in pos:
        pos[node] = (pos[node][0] * width / xmax, pos[node][1])
    return pos