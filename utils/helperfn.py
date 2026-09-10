import os
import numpy as np
from sympy import Matrix
from itertools import product, islice

def msg_remap(x):
    xbar = x.copy()
    xbar[xbar < 0] = 0; xbar[x == 0] = -1
    return xbar

def message_structure(H):
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

def erasure_sampler(codeword, smpsze, wt):
    n = len(codeword)

    if wt is None:
        for _ in range(smpsze):
            mask = np.random.rand(n) < 0.5
            y = codeword.copy(); y[mask] = 0
            yield y.astype(int)
    else:
        for _ in range(smpsze):
            # uniformely sampling wt distance indices
            mask = np.random.choice(n, size=wt, replace=False)
            y = codeword.copy(); y[mask] = 0
            yield y.astype(int)

def find_nontrivial_codeword(H):
    H_sym = Matrix(H.tolist())       # Convert to sympy Matrix
    nullspace = H_sym.nullspace()    # List of basis vectors over rationals

    if len(nullspace) == 0:
        print("No non-trivial codeword exists (full rank).")
        return None

    codeword = np.array(nullspace[0]) % 2
    codeword = codeword.reshape(1, -1)[0]
    codeword[codeword == 0] = -1
    return codeword.astype(int)

def batcher(iterator, batch_size):
    """Yield successive batches from an iterator."""
    iterator = iter(iterator)
    while True:
        batch = list(islice(iterator, batch_size))
        if not batch:
            break
        yield batch

def ptrn2outstr(bits_np):
    # map bits to F3 symbols: -1/0/1 --> 0/2/1 (2 for erasures)
    symbval = bits_np.copy()
    symbval = msg_remap(symbval)
    symbval[symbval < 0] = 2
    symbstr = "".join(map(str, symbval))
    return symbstr

def outstr2ptrn(filepath):
    symbstr = os.path.splitext(os.path.basename(filepath))[0]
    symbval = list(map(int, symbstr.split("_")[2]))
    symbval = np.array(symbval)

    symbval[symbval == 2] = -1      # 0/1/2 --> 0/1/-1
    symbval = msg_remap(symbval)    # 0/1/-1 --> -1/1/0
    return symbval

def nCr(n: int, r: int) -> int:
    r = min(r, n - r)
    return int(np.math.factorial(n) // (np.math.factorial(r) * np.math.factorial(n - r)))

def capped_nCr(n, r, cap=4096):
    r = min(r, n - r)  # symmetry
    result = 1
    
    for k in range(1, r + 1):
        result = result * (n - r + k) // k
        if result > cap:
            return cap  
    return result

def createdir(**kwargs):
    ## creating the directories
    for path in kwargs.values():
        os.makedirs(path, exist_ok=True)
    return