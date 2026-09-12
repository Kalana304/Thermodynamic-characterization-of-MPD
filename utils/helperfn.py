#################################################################################
## Author       : Kalana G Abeywardena
## Created on   : Nov 2025
## Last edited  : Sept 2026
## Purpose      : defines some helper functions useful for other scripts
#################################################################################

import os
import numpy as np
from sympy import Matrix
from itertools import islice

def msg_remap(x):
    """
        Simple remap of symbols that are useful for BEC. We interchange
        0 <-> -1 depending on the required representation for logical bit 0
        and erasures.
    """
    xbar = x.copy()
    xbar[xbar < 0] = 0; xbar[x == 0] = -1
    return xbar

def message_structure(H):
    """
        Defines the graph structure of the code considering its
        Tanner graph.

        args:
            H (numpy.ndarray)   : parity check matrix of size m x n
        
        return:
            var_to_edges (nested list)      : memory locations of edge messages for each var node
            check_to_edges (nested list)    : memory locations of edge messages for each check node
            n_edges                         : total no. of edges
    """
    m, n = H.shape

    # Build edge list
    edge_pos = np.argwhere(H == 1); n_edges = len(edge_pos)
    edge_list = edge_pos  # each row = (check, var) --> ID

    # Build adjacency lists
    var_to_edges = [[] for _ in range(n)]
    check_to_edges = [[] for _ in range(m)]
    for eid, (c, v) in enumerate(edge_list):
        var_to_edges[v].append(eid + n)
        check_to_edges[c].append(eid + n)

    return var_to_edges, check_to_edges, n_edges

def ptrn2outstr(bits_np):
    # map bits to F3 symbols: -1/0/1 --> 0/2/1 (2 for erasures)
    # use this representation as a file name to save trajectories
    symbval = bits_np.copy()
    symbval = msg_remap(symbval)
    symbval[symbval < 0] = 2
    symbstr = "".join(map(str, symbval))
    return symbstr

def outstr2ptrn(filepath):
    # given a filename with a patetrn value embeded, extract the pattern
    symbstr = os.path.splitext(os.path.basename(filepath))[0]
    symbval = list(map(int, symbstr.split("_")[2]))
    symbval = np.array(symbval)

    symbval[symbval == 2] = -1      # 0/1/2 --> 0/1/-1
    symbval = msg_remap(symbval)    # 0/1/-1 --> -1/1/0
    return symbval

def createdir(**kwargs):
    ## creating the directories
    for path in kwargs.values():
        os.makedirs(path, exist_ok=True)
    return