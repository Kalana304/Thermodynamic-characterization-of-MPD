#################################################################################
## Author       : Kalana G Abeywardena
## Created on   : Nov 2025
## Last edited  : Sept 2026
## Purpose      : contains helper functions for state space definition, state 
##                transition matrices, and partial mismatch cost computations.
#################################################################################
import os
import time
import pickle
import numpy as np
from itertools import product
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed

from scipy.special import rel_entr
from scipy.sparse import lil_matrix, save_npz, load_npz

_input2idx = None
_kappa = None
_statesze = None
_nsplen = {"main": 3, "check": 2, "variable": 4}

def init_worker(dictmap=None, submatsze=None, statesze=None):
    global _input2idx, _row_sum, _kappa, _statesze

    _input2idx = dictmap
    _kappa = submatsze
    _statesze = statesze

# read pickle files that has trajectories
def readdata(filepath, TYPE="hash"):
    """
        This generator is written to read line-by-line written trajectories 
        on to .pkl files when running createtrj() function to avoid workers
        having to manage large arrays in memory. 

        args:
            filepath (str)  : filepath to a specific trajectory file
            TYPE (str)      : type of information to read. OPT: {hash, states}

        return:
            yields trajectory information 
    """
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

###########################################################################################################
## +++++++++++++++++++++++++++++++++ setting up the reached state space ++++++++++++++++++++++++++++++++ ##
###########################################################################################################

def load_trjdict(filepath):
    """
        This loads the accessed_dict and init_dict associated with a specific
        trajectory file, specificed by the input filepath.

        args:
            filepath (str)  : filepath to specific trajectory file
        
        return (tuple)      : (accessed_dict, init_dict)
    """
    _readfile = filepath.replace(".pkl", "_dicts.pkl")  # include "_dict" to input filepath
    with open(_readfile, "rb") as f:
        return pickle.load(f)
    
def createstatespace(_patternpaths, _accesseddict_path, _initdict_path, nworkers):
    """
        This function merges all individual accessed_dicts and init_dicts saved for each 
        input pattern to define the reached state space and the initial states of the 
        simulated computation.

        args:
            _patternpaths (list)    : a list of paths to each trajectory file (ends w/ .pkl)
            _accesseddict_path (str): path to save merged accessed state dictionary
            _initdict_path (str)    : path to save merged initial state dictionary 
            nworkers (int)          : number of parallel workers
        
        return:
            accessed_dict (dict)    : merged dictionary of reached state space with (hash, state) as key-val pair
            init_dict (dict)        : merged dictionary of initial states with (hash, state) as key-val pair
    """
    if os.path.isfile(_accesseddict_path) and os.path.isfile(_initdict_path):
        # if dictionaries exist, simply load them --> no need to remerge them.
        accessed_dict = pickle.load(open(_accesseddict_path, 'rb'))
        init_dict = pickle.load(open(_initdict_path, 'rb')) 
    else:
        accessed_dict = dict()
        init_dict = dict()

        with ProcessPoolExecutor(max_workers=nworkers) as ex:
            futures = [ex.submit(load_trjdict, fp) for fp in _patternpaths] # uses parallel processing to load dictionaries for multiple patterns

            for fut in as_completed(futures):
                # merging loaded dictionaries
                _accesseddict, _initdict = fut.result()
                accessed_dict.update(_accesseddict)
                init_dict.update(_initdict)

        # writing dictionaries back --> for clusters + large state spaces, this takes sig. higher time!
        pickle.dump(accessed_dict, open(_accesseddict_path, 'wb'), protocol=pickle.HIGHEST_PROTOCOL)
        pickle.dump(init_dict, open(_initdict_path, 'wb'), protocol=pickle.HIGHEST_PROTOCOL)
        
    print(f"accessed no. of states = {len(accessed_dict)} | initial no. of states = {len(init_dict)}")
    return accessed_dict, init_dict

def createstate2idx(_accesseddict, _lookuppath):
    """
        This function computes a bijection that maps a hash/state value to an index to vectorize 
        probability functions + construct state transition matrices. The bijection used here is a 
        simple lookup table, saved as dictionaries with (hash, idx) as key-value pairs. 

        Procedure:
            (1) using merged accessed dict, states are sorted lexicographically. we want to sort states such that
                [a, b, c] < [a, b, d] < [a, c, ...] < [b, ...]. Highly time consuming when used with np.lexsort().
                Implemented a faster method to do the lexsorting.                
            (2) sort the hash values using the same permutation used to sort states.
            (3) create dictionary with hash values as keys and their indices based on sorted array.

        arsg:
            _accesseddict (dict)    : merged, accessed state space defined as by (hash, state) key-val pairs
            _lookuppath (str)       : path to save the bijection mapping

        return:
            state_to_idx (numpy.ndarry)    : set of sorted states as a stacked ndarray with row no. indicating index
            hash_to_idx (dict)             : (hash, index) key-val pairs for easy lookup
    """

    if os.path.isfile(_lookuppath):
        print("Lookup table found! Loading hash to state idx...")
        with open(_lookuppath, "rb") as f:
            state_to_idx, hash_to_idx = pickle.load(f)

    else:
        print("Lookup table not found! Creating one....")

        # --- FAST extraction ---
        access_hashes = np.fromiter(_accesseddict.keys(), dtype=object) # hash values as a numpy array (uses object to avoid any overflow/precision issues in hash values)
        access_states = np.vstack(list(_accesseddict.values()))         # states are converted to 2D Numpy arrays
        access_states = np.ascontiguousarray(access_states)             # ensures the data are stored contiguously in memory, making view/sorting operations more efficient.

        # FAST LEXICOGRAPHIC SORT (replaces np.lexsort)
        # reinterpret each row of access_states as a single structured element containing all columns as fields. 
        # [x0, x1, x2, ...]  ->  (f0=x0, f1=x1, f2=x2, ...)
        # avoids explicitly constructing and repeatedly handling individual sorting keys/columns
        dtype = np.dtype([
                            (f"f{i}", access_states.dtype)
                            for i in range(access_states.shape[1])
                        ])

        structured = access_states.view(dtype).reshape(-1)      # structured fields are compared in order (f0, then f1, then f2, ...) same as lex row-wise sorting
        sorted_idx = np.argsort(structured, kind="mergesort")   # rows that compare equal retain their original relative ordering with `mergesort`.

        access_states = access_states[sorted_idx]
        access_hashes = access_hashes[sorted_idx]

        # efficient tuple-based dictionary construction
        state_to_idx = access_states.copy() 
        hash_to_idx = dict(zip(access_hashes, range(len(access_hashes))))   # still could be slower for larger state spaces

        with open(_lookuppath, "wb") as f:
            pickle.dump((state_to_idx, hash_to_idx), f, protocol=pickle.HIGHEST_PROTOCOL)

    return state_to_idx, hash_to_idx

def creatensplookupdir(state_to_idx, _nsplookuppath, _machine = "main"):
    """
        Similar to createstate2idx() but we only consider the non-special state variables.
        The states are already sorted following createstate2idx() so no lexsorting is done.
        Instead, reindexing will be done based on only the non-special state variables, which 
        is easy computation.

        args:
            state_to_idx (numpy.ndarray): a numpy 2D array with each row a state of machine, and column a state variable
            _nsplookuppath (str)        : path to save the lookup table 
            _machine (str)                 : which machine is considered. OPT = {main, check, variable} (np. of non-special var vary)

        return:
            nsp_state2idx (dict)        : dictionary with (non-special state, index) as key-val pair  
    """
    if os.path.isfile(_nsplookuppath):
        print("Lookup table for non-special found! Loading hash to state idx...")
        nsp_state2idx = pickle.load(open(_nsplookuppath, "rb"))
    else:
        print("Creating lookup table for non-special state var ...")
        nsp_state2idx = {}
        nsplen = _nsplen[_machine]             # no. of non-special state var for each machine
        for full_state in state_to_idx:
            nsp_state = tuple(full_state[:nsplen])  # get the non-special state vars as a tuple
            if len(nsp_state2idx) == 0:
                nsp_state2idx[nsp_state] = 0        # if dict is empty, assign index 0
                continue

            if nsp_state not in nsp_state2idx.keys():   # if dict is non-empty but tuple is not in, assign len of dict as its index
                nsp_state2idx[nsp_state] = len(nsp_state2idx)

        with open(_nsplookuppath, "wb") as f:
            pickle.dump(nsp_state2idx, f, protocol=pickle.HIGHEST_PROTOCOL)

    return nsp_state2idx

###########################################################################################################
## +++++++++++++++++++++++++++ (stochastic) state transition matrix creation +++++++++++++++++++++++++++ ##
###########################################################################################################
def initstochmatrix(_sze, _nsub_mats, _submat_size, _stochpath, _transpath):
    """
        This initialize sub-matrices and directories to save the transitions 
        that will be used to update each sub-matrix later.

        args:
            _sze (int)          : size of the state space (or full transition matrix)
            _nsub_mats (int)    : no. of submatrices to save/create
            _submat_size (int)  : size of each submatrix size 
            _stochpath (str)    : path to save the initialized matrices
            _transpath (str)    : path to save the transitions (here we create sub-dir with sub-matrix name)
        
        return:
            none
    """
    # initialize transition matrices
    n_sub = 0; disp_freq = int(min(500, _nsub_mats))

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
    """
        This function processes each trajectory to determine the state transition pairs that will be used to update the
        main state transition matrix. The processing will align with the submatrices initialized with initstochmatrix().
        Every transition pair, that belongs to a specific submatrix will be saved to the sub-directories created as .pkl
        files. 

        args:
            filepath (str)      : filepath to a specific trajectory file of a pattern processed
            out_subdir (str)    : directory where the transitions are saved
        
        return:
            none
    """
    data = readdata(filepath, TYPE = "hash")    # gets the trajectory information as lists of hash values
    updates = defaultdict(list)

    for seq in data:
        # each seq is a trajectory that has started with the processed input pattern
        for u in range(len(seq)):
            n1 = _input2idx.get(seq[u])                                 # hash2idx lookup (init across all workers) to get idx of state i    
            n2 = _input2idx.get(seq[u+1]) if u < len(seq)-1 else n1     # hash2idx lookup (init across all workers) to get idx of state i + 1

            # indexing for submatrix 
            out_start = (n2 // _kappa) * _kappa                         
            out_end = min(out_start + _kappa - 1, _statesze - 1)
            in_start  = (n1 // _kappa) * _kappa
            in_end  = min(in_start  + _kappa - 1, _statesze - 1)      

            n1_bar = n1 - in_start      # row index within the submatrix 
            n2_bar = n2 - out_start     # column index within the submatrix

            matrix_name = f"matrix_{out_start}_{out_end}_{in_start}_{in_end}"
            updates[matrix_name].append((n1_bar, n2_bar))   # for each submatrix name, accumulate transition pairs (memory extensive for large state spaces)

    # save local updates
    for matrix_name, transitions in updates.items():
        out_file = os.path.join(out_subdir, matrix_name, os.path.basename(filepath))
        with open(out_file, "wb") as f:
            pickle.dump(transitions, f, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"Finished processing {filepath}.")

def collect_transitions_nsp(filepath, out_subdir, machine="main"):
    """
        Same as collect_transitions() but collects the transtion pairs, init and final, for 
        non-special variable transitions. This will be used to obtain the initial distribution
        for internal variables, as we assume they are initialized by the terminal values. 

        args:
            filepath (str)      : filepath to a specific trajectory file of a pattern processed
            out_subdir (str)    : directory where the transitions are saved
            machine (str)       : which machine is being processed
        
        return:
            none
    """
    updates = defaultdict(list)
    nsplen = _nsplen[machine]   # get no. of non-special state vars for each machine
    for (start_state, end_state) in readdata(filepath, TYPE = "states"):
        n1 = _input2idx.get(start_state[ : nsplen])
        n2 = _input2idx.get(end_state[ : nsplen])

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
    """
        This function updates the submatrices initialized in initstochmatrix() based on the transitions 
        saved using collect_transitions() or  collect_transitions_nsp(). This will only update the 
        transition counts from an input state (row) to an output state (column) of a specific submatrix.

        args:
            mat_trans_path (str)    : sub-dir where state transitions files are saved for a specific submatrix
            matrix_path (str)       : directory where all submatrices are saved
        
        return:
            none
    """
    # update state transition matrices
    _matgen = [_mat.path for _mat in os.scandir(mat_trans_path) if _mat.name.endswith(".pkl")]  # get all transition files in the subdir
    _matname = os.path.basename(mat_trans_path) + ".npz"                                        # define the submatrix name
    sub_mat = load_npz(os.path.join(matrix_path, _matname)).tolil()                             # load the submatrix 

    for _matpath in _matgen:
        pairs = pickle.load(open(_matpath, "rb"))   # read the state transition file 

        for n_pair in pairs:
            sub_mat[n_pair] += 1    # update the transition counts in the loaded submatrix.

    save_npz(os.path.join(matrix_path, _matname), sub_mat.tocsr())      # write the submatrix back to harddrive
    print(f"Finished updating {_matname} --> {len(_matgen)} files")

def normalizestochmap(sze, submat_size, _path, _rowsum = None, _normalize = False):
    """
        This function convertes the row transition counts to probabilities, column-wise i.e., 
        p_{ij} = c_{ij} / \sum_{i} c_{ij} for each (i, j) entry of the state transition matrix.

        args:
            sze (int)               : full state space size (accessed or non-special states)
            submat_size (int)       : submatrix size
            _path (str)             : sub-dir where the submatrices are saved
            _rowsum (numpy.array)   : a numpy array to pass row-wise (technically col-wise) total transition counts when _normalize = True
            _normalize (bool)       : either to normalize the matrix (if row-wise total transitions are obtained) or to obtain row-wise total transitions
        
        return:
            row_sums_full
    """
    # column-wise normalization of transition matrices
    row_sums_full = np.zeros(sze); n_sub = 0
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
                # if _normalize and a total row-wise counts are given, get transition probabilities
                sub_mat = sub_mat.tocsr()
                sub_mat = sub_mat.multiply(1 / _rowsum[i : i + row_size, np.newaxis]).tocsr()
                save_npz(os.path.join(_path, matrix_name), sub_mat.tocsr())

            row_sums = np.array(sub_mat.sum(axis=1)).flatten()
            n_ent = len(row_sums)
            row_sums_full[i : i + n_ent] += row_sums        # running summations of row-wise transition counts from each submatrix

    if not(_normalize):
        row_sums_full[row_sums_full == 0] = 1.0             # if not _normalize, fill rows without no transitions with 1 (mostly for non=scpecial transitions, with conditioned spaces)

    print(f"min row sum = {np.min(row_sums_full)} and max row sum = {np.max(row_sums_full)}")
    return row_sums_full


###########################################################################################################
## +++++++++++++++++++++++++++++++++ periodic mismatch cost computation ++++++++++++++++++++++++++++++++ ##
###########################################################################################################

def blockprobcompute(prev_dist, _sze, _submat_size, _path):
    """
        This function performs bloack-wise probability distribution computation using submatrices. For 
        mismatch cost computations, this acts as a bottleneck as the computation is done seriallized manner. 

        args:
            prev_dist (numpy.array)     : prob. vector from previous clock cycle
            _sze (int)                  : length of the prob. vector
            _submat_size (int)          : size of the submatrix
            _path (str)                 : path to submatrices
        
        return:
            post_dist (numpy.array)     : prob. vector after current clock cycle
    """
    post_dist = np.zeros_like(prev_dist)
    
    for j in range(0, _sze, _submat_size):
        col_size = min(_submat_size, _sze - j)
        for i in range(0, _sze, _submat_size):
            row_size = min(_submat_size, _sze - i)
            _out_matrix = f"matrix_{j}_{j + col_size - 1}_{i}_{i + row_size - 1}.npz"

            if not os.path.exists(os.path.join(_path, _out_matrix)):
                continue
            sub_mat = load_npz(os.path.join(_path, _out_matrix))                    # load submatrix
            post_dist[j : j + col_size] += sub_mat.T @ prev_dist[i : i + row_size]  # block-wise mat-mul
    
    return post_dist

# partial mmc computation (main decoder machine)
def pmmc(alpha, epsilon, _initidx, _nsp_state2idx, _codewords, dirargs, kwargs, _erroridx = None):
    """
        This function computes the periodic mismatch cost for each clock cycle for main decoder machine.

        args:
            alpha (float)           : message bit bias
            epsilon (float)         : erasure prob. for a given BEC channel
            _initidx (numpy.array)  : an array of indices of initial states used for decoding
            _nsp_state2idx (dict)   : lookup table for non-special state variables
            _codewords (numpy.array): an array of indices of codewords used (to align lexsorted order)
            dirargs (dict)          : dictionary of relevant directory paths
            kwargs (dict)           : keyword arguments (from argparser)
            _erroridx (numpy.array) : if not None, condition input distribution on these error indices
        
        return:
            none
    """
    ## prior distribution definition
    if not kwargs.optq0:
        q0 = (1 / kwargs.sze) * np.ones(kwargs.sze)                     # if no specific prior is given, use uniform dist. over state space
    else:
        q0file_ = os.path.join(kwargs.projroot, f"data/q0nunif_main.npz")   # if a different prior is to be used, either load it, or optimize and use it

        if os.path.isfile(q0file_):
            print(f"Found optimized q0 --> Loading from {q0file_}...")
            q0 = np.load(q0file_)["q0_dist"]
        else:
            _, q0 = find_q0(dirargs = dirargs, kwargs = kwargs)
            np.savez_compressed(q0file_, q0_dist = q0)

    q1 = blockprobcompute(q0, kwargs.sze, kwargs.kappa, dirargs.get("stochmat_dir"))    # for a given prior q0, get distribution after one clock cycle using blockmat computation

    ## creating prob distribution induced by the channel (input distribution)
    p_in_channel = np.zeros(2**kwargs.n)    # input to the channel
    p_in_enc = [alpha, 1 - alpha]           # message bit bias

    for _ in range(kwargs.n - kwargs.m - 1):
        p_in_enc = np.outer(p_in_enc, [alpha, 1 - alpha]).flatten(order = 'F')  # dist. of message vectors (and the codewords)

    p_in_channel[_codewords] = p_in_enc                     # define the input prob. to channel with correct codeword ordering
    p_y_x = BECChannel(kwargs.n, erasure_prob = epsilon)    # initialize the channel 
    p_out_channel = p_y_x @ p_in_channel                    # induced output prob. dist. from the BEC channel

    if kwargs.condpin and _erroridx != None:
        _tempdist = np.zeros_like(p_out_channel)
        _tempdist[_erroridx] = p_out_channel[_erroridx]
        p_out_channel = _tempdist / np.sum(_tempdist) if np.sum(_tempdist) != 0 else _tempdist
        del _tempdist

    # creating prob distribution for non-input state variables (m, mu_0)
    nin_size = kwargs.dmax + 1
    p_tnin_decoder = (1 / 3**nin_size)*np.ones(3**nin_size) # say we first start with any value for internal variables
          
    nspidxlist = defaultdict(list)  # get idxs for each nin var combination for marginalization
    for _state, _idx in _nsp_state2idx.items():
        nspidxlist[_state[1:]].append(_idx)

    _p_joint = np.outer(p_out_channel, p_tnin_decoder).flatten(order = "C")     # initial dist. of states without non-special state variables
    p_joint = blockprobcompute(_p_joint, kwargs.nspsze, kwargs.nspkappa, dirargs.get("nspmat_dir")) # final/terminal dist. of states without non-special state variables
    assert len(p_joint) == 3**(kwargs.n + nin_size), f"sizes do not match :: {len(p_joint)} != {3**(kwargs.n + nin_size)}"

    p_marg_nin = np.zeros(3**nin_size)
    for ii, (_state, _idx) in enumerate(nspidxlist.items()):
        p_marg_nin[ii] = np.sum(p_joint[_idx])   # marginalize over input state variables to get the initial dist. for internal state varibles

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

# partial mmc computation (subroutine machine)
def pmmc_subr(_initidx, _nsp_state2idx, dirargs, kwargs):
    """
        Similar to pmmc(), this computes the periodic mismatch cost for each clock cycle for subroutines.
        Since the input to these local devices are not dependent on the channel characteristics, (alpha, epsilon)
        is not needed. 

        args:
            _initidx (numpy.array)  : an array of indices of initial states used in each subroutine
            _nsp_state2idx (dict)   : lookup table for non-special state variables
            dirargs (dict)          : dictionary of relevant directory paths
            kwargs (dict)           : keyword arguments (from argparser)
        
        return:
            none
    """
    ## prior distribution definition
    if not kwargs.optq0:
        q0 = (1 / kwargs.sze) * np.ones(kwargs.sze)                     # if no specific prior is given, use uniform dist. over state space
    else:
        q0file_ = os.path.join(kwargs.projroot, f"data/q0nunif_{kwargs.simname.split('_')[0]}.npz")   # if a different prior is to be used, either load it, or optimize and use it

        if os.path.isfile(q0file_):
            print(f"Found optimized q0 --> Loading from {q0file_}...")
            q0 = np.load(q0file_)["q0_dist"]
        else:
            _, q0 = find_q0(dirargs = dirargs, kwargs = kwargs)
            np.savez_compressed(q0file_, q0_dist = q0)

    q1 = blockprobcompute(q0, kwargs.sze, kwargs.kappa, dirargs.get("stochmat_dir")) # for a given prior q0, get distribution after one clock cycle using blockmat computation


    ## creating input prob distribution for the subroutine:: m_{in, c} for check (m_{in, v}, mu0, wl0) for variable
    _instatelen = kwargs.dmax if "check" in kwargs.simname else kwargs.dmax + 1     # define input reg length
    pin_subr = (1 / 3**_instatelen) * np.ones (3**_instatelen)                      # assumes uniform dist. for edge (+ channel obs) messages

    if "variable" in kwargs.simname:
        pin_wl = np.array([1/3, 2/3])   # for gallager algo. we use wl = {1, 2} --> 1 for l=1, and 2 for l=2,3 (only for variable subroutine message updates)
        pin_subr = np.outer(pin_subr, pin_wl).flatten(order = 'F')

    # creating prob distribution for non-input state variables (m_{out, v/c})
    nin_size = kwargs.dmax
    p_nin_subr = (1 / 3**nin_size) * np.ones(3**nin_size)   

    nspidxlist = defaultdict(list)  # get idxs for each nin var combination for marginalization
    for _state, _idx in _nsp_state2idx.items():
        defdictidx = _state[1] if "check" in kwargs.simname else _state[3]
        nspidxlist[defdictidx].append(_idx)

    _p_joint = np.outer(pin_subr, p_nin_subr).flatten(order = "F")
    p_joint = blockprobcompute(_p_joint, kwargs.nspsze, kwargs.nspkappa, dirargs.get("nspmat_dir"))

    if "check" in kwargs.simname:
        assert len(p_joint) == 3**(_instatelen + nin_size), f"sizes do not match :: {len(p_joint)} != {3**(_instatelen + nin_size)}"
    else:
        assert len(p_joint) == 3**(_instatelen + nin_size) * 2, f"sizes do not match :: {len(p_joint)} != {3**(_instatelen + nin_size) * 2}"

    p_marg_nin = np.zeros(3**nin_size)
    for ii, (_state, _idx) in enumerate(nspidxlist.items()):
        p_marg_nin[ii] = np.sum(p_joint[_idx])   

    # initializing distribution
    p0 = np.zeros(kwargs.sze)
    p0[_initidx] = np.outer(pin_subr, p_marg_nin).flatten(order="F")
    p_prev = p0.copy()

    # computing mismatch cost
    PMCi = np.zeros(kwargs.mmc_iter)

    tic = time.time()
    for _n in range(kwargs.mmc_iter):
        p1 = blockprobcompute(p_prev, kwargs.sze, kwargs.kappa, dirargs.get("stochmat_dir"))

        if (_n % 4 == 0):
            print(f"Iter {_n} --> Total time elapse: {(time.time() - tic):.3f} s --> {np.sum(p1)}")

        PMCi[_n] = np.sum(rel_entr(p_prev, q0)) - np.sum(rel_entr(p1, q1))
        p_prev = p1.copy()

    filename_ = f"mmc_{kwargs.simname.split('_')[0]}sub.npz"
    np.savez_compressed(os.path.join(dirargs.get('mmc_dir'), filename_), mmc = PMCi) 


###########################################################################################################
## +++++++++++++++++++++++++++++++ mismatch cost related misc functions ++++++++++++++++++++++++++++++++ ##
###########################################################################################################

def entropy(p):
    # defines Shannon entropy for a given distribution p(x)
    p = np.clip(p, 1e-12, 1.0)  # Avoid log(0)
    return -np.sum(p * np.log(p))

def objective(q, fx, sze, _submat_sze, matrix_dir):
    # optimization objective for prior finding for a given f(x)
    Aq = blockprobcompute(q, sze, _submat_sze, matrix_dir)
    return entropy(Aq)- entropy(q) + np.dot(fx, q)

def find_q0(dirargs, kwargs, fx = None):
    """
        This code is used to find the optimum prior distribution for a given f(x) entropy
        flow fucntion. Adopted from Yadav et al. (2025) and Wolpert et al. (2019).

        args:
            dirargs (dict)      : dictionary with relevant paths
            kwargs (dict)       : arg parse arguments
            fx (numpy.array)    : entropy flow array

        return:
            cost0 (float)       : entropy production cost for selected prior
            qx (numpy.array)    : prior distribution    
    """
    rng = np.random.RandomState(kwargs.seed)

    q = rng.rand(kwargs.sze); q = q / np.sum(q) # start from some random distribution
    qx = q.copy()

    if fx == None:
        fx = np.ones(kwargs.sze)       # define a uniform entropy flow if not defined

    # entropy produ. cost with previous choice of q
    cost0 = objective(
                      q = q, fx = fx, 
                      sze = kwargs.sze, _sub_sze = kwargs.kappa, 
                      matrix_dir = dirargs.get("stochmat_dir")
                      )  
    
    T = kwargs.T0             # temperature
    s = kwargs.s0             # interpolation para

    for ii in range(kwargs.opt_iter):
        r = rng.rand(kwargs.sze); r = r / np.sum(r)     # take another random prob. distribution
        q_new = (1 - s) * r + s * q                     # create a new candidate (convex combination)
        q_new = q_new / np.sum(q_new)                   # normalize if necessary (not really)

        cost = objective(
                            q = q_new, fx = fx, 
                            sze = kwargs.sze, _sub_sze = kwargs.kappa, 
                            matrix_dir = dirargs.get("stochmat_dir")
                        )

        if (cost < cost0) and (np.random.rand() < np.exp(-(cost - cost0) / T)):
            # if cost is less or if it is chosen with higher prob. update
            q = q_new; qx = q.copy()
            cost0 = cost

        T *= kwargs.gamma         # update T = \gamma T
        s = max(0.9 * s, 0.05)    # reduce cooling para.
        if (ii % 5000 == 0):
            print(f"Running iteration {ii + 1} / {kwargs.opt_iter} --> Cost = {cost0}")
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
                if y[bit_idx] == -1:            # Erasure
                    prob *= erasure_prob
                elif y[bit_idx] == x[bit_idx]:  # Correct reception
                    prob *= (1 - erasure_prob)
                else:                           # Different bit (impossible in BEC)
                    prob = 0
                    break

            transition_matrix[i, j] = prob
    return transition_matrix

