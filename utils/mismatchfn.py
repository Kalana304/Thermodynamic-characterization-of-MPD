import os
import time
import pickle
import numpy as np
from itertools import product
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed

from scipy.special import rel_entr
from scipy.sparse import lil_matrix, save_npz, load_npz, dok_matrix, coo_matrix

from utils.helperfn import outstr2ptrn

_input2idx = None
_kappa = None
_statesze = None

def init_worker(dictmap=None, submatsze=None, statesze=None):
    global _input2idx, _row_sum, _kappa, _statesze

    _input2idx = dictmap
    _kappa = submatsze
    _statesze = statesze

# data
def readdata(filepath, TYPE="hash"):
    with open(filepath, "rb") as f:
        while True:
            try:
                batch = pickle.load(f)
                for item in batch:
                    if TYPE == "hash":
                        yield item[0]
                    elif TYPE == "states":
                        yield item[1], item[2]   # s0, sT
            except EOFError:
                break

def pathstr2base3(smp_paths, n, mod_):
    vy = 3 ** np.arange(n, dtype=object)
    y3base = np.zeros((len(smp_paths)))
    for i, fp in enumerate(smp_paths):
        y3base[i] = np.sum(np.dot(vy, outstr2ptrn(fp)) % mod_, dtype=int) if mod_ != None else np.sum(np.dot(vy, outstr2ptrn(fp)), dtype=int)
    return y3base.astype('int')

# state space define
def load_trjdict(filepath):
    _readfile = filepath.replace(".pkl", "_dicts.pkl")
    with open(_readfile, "rb") as f:
        return pickle.load(f)
    
def createstatespace(_errfiles, _accesseddict_path, _initdict_path, nworkers):
    if os.path.isfile(_accesseddict_path) and os.path.isfile(_initdict_path):
        accessed_dict = pickle.load(open(_accesseddict_path, 'rb'))
        init_dict = pickle.load(open(_initdict_path, 'rb')) 
    else:
        accessed_dict = dict()
        init_dict = dict()

        with ProcessPoolExecutor(max_workers=nworkers) as ex:
            futures = [ex.submit(load_trjdict, fp) for fp in _errfiles]

            for fut in as_completed(futures):
                _accesseddict, _initdict = fut.result()
                accessed_dict.update(_accesseddict)
                init_dict.update(_initdict)

        pickle.dump(accessed_dict, open(_accesseddict_path, 'wb'), protocol=pickle.HIGHEST_PROTOCOL)
        pickle.dump(init_dict, open(_initdict_path, 'wb'), protocol=pickle.HIGHEST_PROTOCOL)
        
    print(f"accessed no. of states = {len(accessed_dict)} | initial no. of states = {len(init_dict)}")
    return accessed_dict, init_dict

def createstate2idx(_accesseddict, _lookuppath):

    if os.path.isfile(_lookuppath):
        print("Lookup table found! Loading hash to state idx...")
        with open(_lookuppath, "rb") as f:
            state_to_idx, hash_to_idx = pickle.load(f)

    else:
        print("Lookup table not found! Creating one....")

        # --- FAST extraction ---
        access_hashes = np.fromiter(_accesseddict.keys(), dtype=object)
        access_states = np.vstack(list(_accesseddict.values()))
        access_states = np.ascontiguousarray(access_states)

        # FAST LEXICOGRAPHIC SORT (replaces np.lexsort)
        dtype = np.dtype([
                            (f"f{i}", access_states.dtype)
                            for i in range(access_states.shape[1])
                        ])

        structured = access_states.view(dtype).reshape(-1)
        sorted_idx = np.argsort(structured, kind="mergesort")

        access_states = access_states[sorted_idx]
        access_hashes = access_hashes[sorted_idx]

        # Efficient tuple-based dictionary construction
        state_to_idx = access_states.copy() 
        hash_to_idx = dict(zip(access_hashes, range(len(access_hashes))))

        with open(_lookuppath, "wb") as f:
            pickle.dump((state_to_idx, hash_to_idx), f, protocol=pickle.HIGHEST_PROTOCOL)

    return state_to_idx, hash_to_idx

def creatensplookupdir(state_to_idx, _nsplookuppath):
    if os.path.isfile(_nsplookuppath):
        print("Lookup table for non-special found! Loading hash to state idx...")
        nsp_state2idx = pickle.load(open(_nsplookuppath, "rb"))

    else:
        print("Creating lookup table for non-special state var ...")
        nsp_state2idx = {}
        for full_state in state_to_idx:
            nsp_state = tuple(full_state[:3])
            if len(nsp_state2idx) == 0:
                nsp_state2idx[nsp_state] = 0
                continue

            if nsp_state not in nsp_state2idx.keys():
                nsp_state2idx[nsp_state] = len(nsp_state2idx)

        with open(_nsplookuppath, "wb") as f:
            pickle.dump(nsp_state2idx, f, protocol=pickle.HIGHEST_PROTOCOL)

    return nsp_state2idx

# stoch matrix build up
def initstochmatrix(_sze, _nsub_mats, _submat_size, _stochpath, _transpath):
    # initialize transition matrices
    n_sub = 0
    disp_freq = int(min(500, _nsub_mats))

    for i in range(0, _sze, _submat_size):
        for j in range(0, _sze, _submat_size):
            n_sub += 1
        
            # Compute actual submatrix size for edges
            row_size = min(_submat_size, _sze - i)  # remaining rows
            col_size = min(_submat_size, _sze - j)  # remaining columns

            # naming matrix_{out_start}_{out_end}_{in_start}_{in_out}.npy
            matrix_name = f"matrix_{j}_{j + col_size - 1}_{i}_{i + row_size - 1}.npz"
            sub_mat = lil_matrix((row_size, col_size))
            save_npz(os.path.join(_stochpath, matrix_name), sub_mat.tocsr())

            _mat_dir_str = matrix_name.split(".npz")[0]
            os.makedirs(os.path.join(_transpath, _mat_dir_str), exist_ok=True)

            if n_sub % disp_freq == 0:
                print(f"Saved {n_sub} / {_nsub_mats**2}")
    return

def collect_transitions(filepath, out_subdir):
    data = readdata(filepath, TYPE = "hash")
    updates = defaultdict(list)
    for seq in data:
        for u in range(len(seq)):
            n1 = _input2idx.get(seq[u])
            n2 = _input2idx.get(seq[u+1]) if u < len(seq)-1 else n1
            
            out_start = (n2 // _kappa) * _kappa
            out_end = min(out_start + _kappa - 1, _statesze - 1)
            in_start  = (n1 // _kappa) * _kappa
            in_end  = min(in_start  + _kappa - 1, _statesze - 1)      

            n1_bar = n1 - in_start
            n2_bar = n2 - out_start

            matrix_name = f"matrix_{out_start}_{out_end}_{in_start}_{in_end}"
            updates[matrix_name].append((n1_bar, n2_bar))

    # save local updates
    for matrix_name, transitions in updates.items():
        out_file = os.path.join(out_subdir, matrix_name, os.path.basename(filepath))
        with open(out_file, "wb") as f:
            pickle.dump(transitions, f, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"Finished processing {filepath}.")

def collect_transitions_nsp(filepath, out_subdir):
    updates = defaultdict(list)

    for (start_state, end_state) in readdata(filepath, TYPE = "states"):
        n1 = _input2idx.get(start_state[ : 3])
        n2 = _input2idx.get(end_state[ : 3])

        out_start = (n2 // _kappa) * _kappa
        out_end = min(out_start + _kappa - 1, _statesze - 1)
        in_start  = (n1 // _kappa) * _kappa
        in_end  = min(in_start  + _kappa - 1, _statesze - 1)      

        n1_bar = n1 - in_start
        n2_bar = n2 - out_start

        matrix_name = f"matrix_{out_start}_{out_end}_{in_start}_{in_end}"
        updates[matrix_name].append((n1_bar, n2_bar))

    # save local updates
    _outstr = os.path.basename(filepath)
    for matrix_name, transitions in updates.items():
        out_file = os.path.join(out_subdir, matrix_name, _outstr)
        with open(out_file, "wb") as f:
            pickle.dump(transitions, f, protocol=pickle.HIGHEST_PROTOCOL)
    
    print(f"Finished processing {_outstr}.")

def update_matrix(mat_trans_path, matrix_path):
    # update state transition matrices
    _matgen = [_mat.path for _mat in os.scandir(mat_trans_path) if _mat.name.endswith(".pkl")]
    _matname = os.path.basename(mat_trans_path) + ".npz"
    sub_mat = load_npz(os.path.join(matrix_path, _matname)).tolil()

    for _matpath in _matgen:
        pairs = pickle.load(open(_matpath, "rb"))

        for n_pair in pairs:
            sub_mat[n_pair] += 1

    save_npz(os.path.join(matrix_path, _matname), sub_mat.tocsr())
    print(f"Finished updating {_matname} --> {len(_matgen)} files")

def normalizestochmap(sze, submat_size, _path, _rowsum = None, _normalize = False):
    # column-wise normalization of transition matrices
    row_sums_full = np.zeros(sze)
    n_sub = 0
    for i in range(0, sze, submat_size):
        for j in range(0, sze, submat_size):
            n_sub += 1

            # Compute actual submatrix size for edges
            row_size = min(submat_size, sze - i)  # remaining rows
            col_size = min(submat_size, sze - j)  # remaining columns

            # naming matrix_{out_start}_{out_end}_{in_start}_{in_out}.npy
            matrix_name = f"matrix_{j}_{j + col_size - 1}_{i}_{i + row_size - 1}.npz"

            sub_mat = load_npz(os.path.join(_path, matrix_name))
            if _normalize and (_rowsum is not None):
                sub_mat = sub_mat.tocsr()
                sub_mat = sub_mat.multiply(1 / _rowsum[i : i + row_size, np.newaxis]).tocsr()
                save_npz(os.path.join(_path, matrix_name), sub_mat.tocsr())

            row_sums = np.array(sub_mat.sum(axis=1)).flatten()
            n_ent = len(row_sums)
            row_sums_full[i : i + n_ent] += row_sums

    # if not(_normalize):
    #     row_sums_full[row_sums_full == 0] = 1.0
    print(f"min row sum = {np.min(row_sums_full)} and max row sum = {np.max(row_sums_full)}")
    return row_sums_full

def blockprobcompute(prev_dist, _sze, _submat_size, _path):
    post_dist = np.zeros_like(prev_dist)
    
    for j in range(0, _sze, _submat_size):
        col_size = min(_submat_size, _sze - j)
        for i in range(0, _sze, _submat_size):
            row_size = min(_submat_size, _sze - i)
            _out_matrix = f"matrix_{j}_{j + col_size - 1}_{i}_{i + row_size - 1}.npz"

            if not os.path.exists(os.path.join(_path, _out_matrix)):
                continue
            sub_mat = load_npz(os.path.join(_path, _out_matrix))
            post_dist[j : j + col_size] += sub_mat.T @ prev_dist[i : i + row_size]
    
    return post_dist


# mismatch cost related misc functions
def entropy(p):
    p = np.clip(p, 1e-12, 1.0)  # Avoid log(0)
    return -np.sum(p * np.log(p))

def objective(q, fx, sze, _submat_sze, matrix_dir):
    Aq = blockprobcompute(q, sze, _submat_sze, matrix_dir)
    return entropy(Aq)- entropy(q) + np.dot(fx, q)

def find_q0(sze, _sub_sze = 3**11, fx = None, matrix_dir = None, max_iters=500000, T0=1.0, alpha=0.999, s0=0.99, state=100):
    rng = np.random.RandomState(state)

    q = np.ones(sze); q = q / np.sum(q)       # start from uniform dist.
    qx = q.copy()

    if fx == None:
        fx = np.ones(sze)       # define a uniform entropy flow if not defined

    cost0 = objective(q, fx, sze, _sub_sze, matrix_dir)  # cost with previous choice of q
    T = T0             # temperature
    s = s0             # interpolation para

    for ii in range(max_iters):
        r = rng.rand(sze)              # sample another random vector
        r = r / np.sum(r)              # normalize to make it a prob. dist.
        q_new = (1 - s) * r + s * q    # create a new candidate
        q_new = q_new / np.sum(q_new)  # normalize if necessary

        cost = objective(q_new, sze, _sub_sze, matrix_dir)       # cost with candidate distribution

        if (cost < cost0) and (np.random.rand() < np.exp(-(cost - cost0) / T)):
            # if cost is less or if it is chosen with higher prob. update
            q = q_new; qx = q.copy()
            cost0 = cost

        T *= alpha               # update T = \alpha T
        s = max(0.9 * s, 0.05)    # reduce cooling para.
        if (ii % 5000 == 0):
            print(f"Running iteration {ii + 1} / {max_iters} --> Cost = {cost0}")
    return cost0, qx

def BECChannel(n, erasure_prob):
    """
    Compute the full state transition from 2^n input codewords to 3^n output vectors.

    Parameters:
    -----------
    n : int
        Length of codewords
    erasure_prob : float
        Erasure probability (epsilon)

    Returns:
    --------
    transition_matrix : numpy.ndarray
        Matrix of shape (3^n, 2^n) where entry [i,j] is P(y_i | x_j)
    """
    
    # Generate all possible inputs (2^n codewords)
    vb = 2**np.arange(n)
    input_vectors = np.array(list(product([0, 1], repeat=n)))
    input_idx = np.dot(input_vectors, vb)
    input_idx = np.argsort(input_idx)

    input_vectors = input_vectors[input_idx]
    num_inputs = 2**n
    assert len(np.unique(input_vectors, axis=1)) == num_inputs

    # Generate all possible outputs (3^n vectors with values in {0,1,2})
    v3 = 3**np.arange(n)
    output_vects_ = np.array(list(product([-1, 0, 1], repeat=n)))
    output_idx = np.dot(output_vects_, v3)
    output_idx = np.argsort(output_idx)

    output_vectors = np.array(list(product([0, -1, 1], repeat=n)))
    output_vectors = output_vectors[output_idx]
    num_outputs = 3**n
    assert len(np.unique(output_vectors, axis=1)) == num_outputs

    # Build transition matrix P(y|x)
    transition_matrix = np.zeros((num_outputs, num_inputs))

    for j, x in enumerate(input_vectors):
        for i, y in enumerate(output_vectors):
            # Compute P(y|x) for BEC
            prob = 1.0
            for bit_idx in range(n):
                if y[bit_idx] == -1:  # Erasure
                    prob *= erasure_prob
                elif y[bit_idx] == x[bit_idx]:  # Correct reception
                    prob *= (1 - erasure_prob)
                else:  # Different bit (impossible in BEC)
                    prob = 0
                    break

            transition_matrix[i, j] = prob
    return transition_matrix

def pmmc(alpha, epsilon, _initidx, _nsp_state2idx, _codewords, dirargs, kwargs):
    ## prior distribution definition
    if not kwargs.optq0:
        q0 = (1 / kwargs.sze) * np.ones(kwargs.sze) 
    else:
        q0file_ = os.path.join(dirargs.get("savedir"), "q0_main.npz")

        if os.path.isfile(q0file_):
            print(f"Found optimized q0 --> Loading from {q0file_}...")
            q0 = np.load(q0file_)["q0_dist"]
        else:
            _, q0 = find_q0(
                            sze = kwargs.sze, _sub_sze = kwargs.kappa, 
                            matrix_dir = dirargs.get("stochmat_dir"), 
                            max_iters = kwargs.opt_iter
                            )
            np.savez_compressed(os.path.join(dirargs.get("savedir"), 'q0_main.npz'), q0_dist = q0)

    q1 = blockprobcompute(q0, kwargs.sze, kwargs.kappa, dirargs.get("stochmat_dir"))

    ## creating prob distribution induced by the channel
    p_in_channel = np.zeros(2**kwargs.n)
    p_in_enc = [alpha, 1 - alpha]

    for h in range(kwargs.n - kwargs.m - 1):
        p_in_enc = np.outer(p_in_enc, [alpha, 1 - alpha]).flatten(order = 'F')

    p_in_channel[_codewords] = p_in_enc
    p_y_x = BECChannel(kwargs.n, erasure_prob = epsilon)
    p_out_channel = p_y_x @ p_in_channel

    # creating prob distribution for non-input state variables (m, mu_0)
    nin_size = kwargs.dmax + 1
    p_tnin_decoder = (1 / 3**nin_size)*np.ones(3**nin_size)
          
    nspidxlist = defaultdict(list)  # get idxs for each nin var combination for marginalization
    for _state, _idx in _nsp_state2idx.items():
        nspidxlist[_state[1:]].append(_idx)

    _p_joint = np.outer(p_out_channel, p_tnin_decoder).flatten(order = "C")
    p_joint = blockprobcompute(_p_joint, kwargs.nspsze, kwargs.nspkappa, dirargs.get("nspmat_dir"))
    assert len(p_joint) == 3**(kwargs.n + nin_size), f"sizes do not match :: {len(p_joint)} != {3**(kwargs.n + nin_size)}"

    p_marg_nin = np.zeros(3**nin_size)
    for ii, (_state, _idx) in enumerate(nspidxlist.items()):
        p_marg_nin[ii] = np.sum(p_joint[_idx])   

    # initializing distribution
    p0 = np.zeros(kwargs.sze)
    p0[_initidx] = np.outer(p_out_channel, p_marg_nin).flatten(order="C")
    p_prev = p0.copy()

    # computing mismatch cost
    PMCi = np.zeros(kwargs.mmc_iter)

    tic = time.time()
    for _n in range(kwargs.mmc_iter):
        p1 = blockprobcompute(p_prev, kwargs.sze, kwargs.kappa, dirargs.get("stochmat_dir"))

        if (_n % 100 == 0):
            print(f"Iter {_n} --> Total time elapse: {(time.time() - tic) / 60:.3f} min --> {np.sum(p1)}")

        PMCi[_n] = np.sum(rel_entr(p_prev, q0)) - np.sum(rel_entr(p1, q1))
        p_prev = p1.copy()
        
    filename_ = f"mmc_alpha_{alpha:.2f}_eps_{epsilon:.2f}.npz"
    np.savez_compressed(os.path.join(dirargs.get('mmc_dir'), filename_), mmc = PMCi) 