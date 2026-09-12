#################################################################################
## Author       : Kalana G Abeywardena
## Created on   : Nov 2025
## Last edited  : Sept 2026
## Purpose      : defines computations carried by main decoder machine to generate 
##                trajectroy files for each possible erasure pattern it would get
#################################################################################

import os
import gc
import pickle
import numpy as np
from itertools import product

from utils.helperfn import msg_remap, ptrn2outstr

## Global variabels (so workers have access to them easily than passing them as arguments)
GLOB_H = None
GLOB_VAR2EDGES = None
GLOB_CHK2EDGES = None
GLOB_NEDGES = None
GLOB_DCMAX = None
GLOB_DVMAX = None
GLOB_LMAX = None
GLOB_NPROG = None
GLOB_BARR = None
GLOB_SAVEDIR = None

MU_STATES = None
MU0_STATES = None
POW3 = None  

def worker_init(args):
    """
        This function initializes the common parameters that will be utilized by
        parallely running workers during the simulation.
    """
    global GLOB_H, GLOB_VAR2EDGES, GLOB_CHK2EDGES
    global GLOB_NEDGES, GLOB_DCMAX, GLOB_DVMAX, GLOB_LMAX, GLOB_NPROG
    global GLOB_BARR, GLOB_SAVEDIR
    global MU_STATES, MU0_STATES, POW3

    # prevent BLAS thread explosion
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["NUMEXPR_NUM_THREADS"] = "1"
    
    GLOB_H = args.H
    GLOB_VAR2EDGES = args.var2edges
    GLOB_CHK2EDGES = args.check2edges
    GLOB_NEDGES = args.nedges
    GLOB_DCMAX = args.dcmax
    GLOB_DVMAX = args.dvmax
    GLOB_LMAX = args.lmax
    GLOB_NPROG = args.nprog
    GLOB_BARR = args.barr
    GLOB_SAVEDIR = args.savedir

    # precompute internal reg values
    MU_STATES = np.array(list(product([-1, 0, 1], repeat=args.dmax)))
    MU0_STATES = np.array(list(product([-1, 0, 1], repeat=1)))

    # precompute powers of 3 for base-3 encoding
    POW3 = args.POW3

def process_pattern(params):
    """Process a single output from BEC chanel"""
    iy, y_, inputsze, flush_sze = params
    try:
        _, n = GLOB_H.shape
        y = np.array(y_)

        accessed_dict = dict()
        init_dict = dict()

        filename = f"statehash_y_{ptrn2outstr(y)}.pkl"
        filepath = os.path.join(GLOB_SAVEDIR, "trajectories", filename)

        message_arr = np.zeros(GLOB_NEDGES + n)

        buffer = []

        # open trajectory file
        traj_file = open(filepath, "ab")

        for m_arr in MU_STATES:
            m_arr = np.array(m_arr)
            for mu0 in MU0_STATES:
                mu0 = np.array(mu0)
                hvals, states = gallager_bec_decoder(
                                                        H = GLOB_H, barr = GLOB_BARR, 
                                                        y = y, mu0 = mu0, m_reg = m_arr, 
                                                        l_max = GLOB_LMAX, message_arr = message_arr, 
                                                        check_to_edges = GLOB_CHK2EDGES, var_to_edges = GLOB_VAR2EDGES, 
                                                        POW3 = POW3
                                                    )
                buffer.append((hvals, states[0], states[-1]))

                for h, s in zip(hvals, states):
                    accessed_dict[h] = s
                
                init_dict[hvals[0]] = states[0]

                # flush periodically
                if len(buffer) >= flush_sze:
                    pickle.dump(buffer, traj_file, protocol=pickle.HIGHEST_PROTOCOL)
                    buffer.clear()
        # final flush
        if buffer:
            pickle.dump(buffer, traj_file, protocol=pickle.HIGHEST_PROTOCOL)
        
        traj_file.close()

        # save dictionaries separately
        dictfile = filepath.replace(".pkl", "_dicts.pkl")
        with open(dictfile, "wb") as f:
            pickle.dump((accessed_dict, init_dict), f)

        print(f"Completed y pattern {iy + 1} / {inputsze}", flush=True)

        del accessed_dict, init_dict
        gc.collect()

    except Exception as e:
        print(f"[Worker crash] iy = {iy} | {e}", flush=True)
        raise

def initialize(mu0, _message_arr, var_to_edges, n):
    """
        Initialize the message array (memory locations) with the initial edge messages and channel observations
        (Not the initialization of any comp. machines). Here we consider the output from channel first initialize 
        the memory. Then the main decoder machine reads the channel observations to intialize its input/output 
        register to start the computation. For the simulation purposes, this will be called within the main decoder program.

        args:
            mu0 (numpy.array)           : channel output saved as channel observatiosn messages in memory
            _message_arr (numpy.array)  : contiguous numpy array that represents a memory  
            var_to_edges (list of list) : defines the mapping where the edge messages are located in the memory
            n (int)                     : block size
        
        return:
            _message_arr (numpy.array)  : initialized memory array
    """
    _message_arr[:n] = mu0.copy()
    for v in range(n):
        nbrs = var_to_edges[v]
        _message_arr[nbrs] = mu0[v]
    return _message_arr

def compute_chk_to_edge(chk_msg_reg, dc):
    """ 
        updates the messages to be sent by a chk node.
    """
    chk_msg_in = chk_msg_reg.copy()
    chk_msg_out = chk_msg_reg.copy()
    for i in range(dc):
        chk_msg_out[i] = np.prod(np.delete(chk_msg_in[:dc], i))

    return chk_msg_out

def compute_var_to_edge(var_msg_reg, dv, mu0_v, wl):
    """ 
        updates the messages to be sent by a var node 
    """
    chk_msg_in = var_msg_reg.copy()
    chk_msg_out = var_msg_reg.copy()

    for i in range(dv):
        chk_msg_out[i] = np.sign(wl * mu0_v + np.sum(np.delete(chk_msg_in[:dv], i)))[-1]

    return chk_msg_out

def update_states(barr, c_hat, m_, mu0, chk_idx, var_idx, l, _s, _p, pc_, hvals, states, POW3):
    """
        Function to update the states and their hash values for each clock cycle.
        
        args:
            barr (numpy.array)      : random values used to hash the state values
            c_hat (numpy.array)     : input/output pattern that comes from channel and subsequently getting updated when decoding.
            m_ (numpy.array)        : internal edge message register intialized by different random values.
            mu0 (int)               : channel observation register intialized by different random values.
            chk_idx (int)           : loop counter for check nodes
            var_idx (int)           : loop counter for var nodes
            l (int)                 : loop counter for message-passing iteration
            _s (bool)               : success flag
            _p (bool)               : progress flag
            pc_ (int)               : program counter for the main decoder machine
            hvals (list)            : a list to which hashvalues are appeneded to keep track of trajectories
            states (list)           : a list to which the state values are appended
            POW3 (str)              : power 3 for state encode
        
        return:
            hvals (list)            : updated list of hash values
            states (list)           : updated list of state values
    """
    n = len(c_hat)
    m_reg_len = len(m_)
    vm = 3**np.arange(m_reg_len)

    hvals.append( np.dot(c_hat, barr[:n]) + np.dot(m_, barr[n : n + m_reg_len]) + 
                  np.dot(mu0, barr[n + m_reg_len : n + m_reg_len + 1]) + 
                   chk_idx * barr[n + m_reg_len + 1] + var_idx * barr[n + m_reg_len + 2] + 
                   l * barr[n + m_reg_len + 3] + _s * barr[n + m_reg_len + 4] +
                   _p * barr[n + m_reg_len + 5] + pc_ * barr[n + m_reg_len + 6]
                )
    
    _y2int = np.sum(np.dot(POW3, c_hat), dtype=int)
    states.append((_y2int, np.sum(np.dot(vm, m_), dtype=int), int(mu0[-1]), chk_idx, var_idx, l, _s, _p, pc_))
    return hvals, states

def gallager_bec_decoder(H, barr, y, mu0, m_reg, l_max, message_arr, check_to_edges, var_to_edges, POW3 = None):
    """
        This function implements the main decoder machine, whose input register gets intialized by the channel output
        and invokes subroutines to update messages iteratively which interacting with the memory. This carries out 
        control operations, information updates to resolve the pattern receieved, assuming the input and output reg 
        are the same. 

        args:
            H (numpy.ndarry)                : parity-check matrix of size m x n
            barr (numpy.array)              : random values used to hash the state values
            y (numpy.array)                 : received erasure pattern from channel
            mu0 (numpy.array)               : channel observation (randomly initialized)
            m_reg (numpy.array)             : edge messages (randomly initialized)
            l_max (int)                     : max. message passing iterations
            message_arr (numpy.array)       : memory array (contains channel observation + edge messages)
            check_to_edges (list of list)   : mapping of edge message neighbourhood for each check node
            var_to_edges (list of list)     : mapping of edge message neighbourhood for each var node
            POW3 (numpy.array)

        return:
            hvals (list)            : list of hash values of the visited states along the computation
            states (list)           : list of state values of the visited states along the computation
    """
    m, n = H.shape          # parameters available to system (consider hardcoded)
    y_hat = y.copy()        # initialize input register with channel output
    mu0 = mu0.copy()        # random value to channel obs. reg initially
    marr = m_reg.copy()     # random values to edge message reg. initially
    l, c, v, _success, _progress = 0, 0, 0, 0, 1    # initialize loop counters, boolean flags
    message_arr = initialize(y_hat, message_arr, var_to_edges, n) # initializing memory (could have been initialized beforehand)
    
    pc_main = 0
    hvals, states = update_states(barr, y_hat, marr, mu0, c, v, l, _success, _progress, pc_main, [], [], POW3)

    for l in range(1, l_max + 1):   # start of the message-passing iterations
        pc_main = 1
        hvals, states = update_states(barr, y_hat, marr, mu0, c, v, l, _success, _progress, pc_main, hvals, states, POW3)

        c = 0; pc_main = 2
        hvals, states = update_states(barr, y_hat, marr, mu0, c, v, l, _success, _progress, pc_main, hvals, states, POW3)

        v = 0; pc_main = 3
        hvals, states = update_states(barr, y_hat, marr, mu0, c, v, l, _success, _progress, pc_main, hvals, states, POW3)

        wl = 1 if l == 1 else 2; pc_main = 4
        hvals, states = update_states(barr, y_hat, marr, mu0, c, v, l, _success, _progress, pc_main, hvals, states, POW3)

        for c in range(1, m + 1):   # looping over check nodes (follows a flooded schedule, but sequentially executed)
            pc_main = 5
            hvals, states = update_states(barr, y_hat, marr, mu0, c, v, l, _success, _progress, pc_main, hvals, states, POW3)

            nbrs = check_to_edges[c - 1]; dc = len(nbrs); pc_main = 6
            hvals, states = update_states(barr, y_hat, marr, mu0, c, v, l, _success, _progress, pc_main, hvals, states, POW3)

            marr[ : dc] = message_arr[nbrs]; pc_main = 7        # reads local messages from memory that come from variable nodes
            hvals, states = update_states(barr, y_hat, marr, mu0, c, v, l, _success, _progress, pc_main, hvals, states, POW3)

            marr = compute_chk_to_edge(marr, dc); pc_main = 8   # updates the local messages to be sent to variable nodes
            hvals, states = update_states(barr, y_hat, marr, mu0, c, v, l, _success, _progress, pc_main, hvals, states, POW3)

            message_arr[nbrs] = marr[ : dc].copy(); pc_main = 9 # write the updated local messages to the memory
            hvals, states = update_states(barr, y_hat, marr, mu0, c, v, l, _success, _progress, pc_main, hvals, states, POW3)
        
        pc_main = 10 
        hvals, states = update_states(barr, y_hat, marr, mu0, c, v, l, _success, _progress, pc_main, hvals, states, POW3)
        for v in range(1, n + 1):   # looping over variable nodes (follows a flooded schedule, but sequentially executed)
            pc_main = 11
            hvals, states = update_states(barr, y_hat, marr, mu0, c, v, l, _success, _progress, pc_main, hvals, states, POW3)

            _progress = 0; pc_main = 12
            hvals, states = update_states(barr, y_hat, marr, mu0, c, v, l, _success, _progress, pc_main, hvals, states, POW3)

            nbrs = var_to_edges[v - 1]; dv = len(nbrs); pc_main = 13    
            hvals, states = update_states(barr, y_hat, marr, mu0, c, v, l, _success, _progress, pc_main, hvals, states, POW3)

            marr[ : dv] = message_arr[nbrs]; pc_main = 14       # reads local messages from memory that come from check nodes
            hvals, states = update_states(barr, y_hat, marr, mu0, c, v, l, _success, _progress, pc_main, hvals, states, POW3)

            mu0 = message_arr[v - 1 : v]; pc_main = 15          # reads channel observation from memory that originally came from channel
            hvals, states = update_states(barr, y_hat, marr, mu0, c, v, l, _success, _progress, pc_main, hvals, states, POW3)
            
            tmp = np.sign(wl * mu0 + np.sum(marr[ : dv])); pc_main = 16     # resolves the v-th symbol of the received pattern (consider all input messages)
            hvals, states = update_states(barr, y_hat, marr, mu0, c, v, l, _success, _progress, pc_main, hvals, states, POW3)

            if tmp != y_hat[v - 1]:     # if the resolved value is different from what we have, update the register!!
                pc_main = 17
                hvals, states = update_states(barr, y_hat, marr, mu0, c, v, l, _success, _progress, pc_main, hvals, states, POW3)

                y_hat[v - 1] = tmp.copy(); pc_main = 18
                hvals, states = update_states(barr, y_hat, marr, mu0, c, v, l, _success, _progress, pc_main, hvals, states, POW3)

                _progress = 1; pc_main = 19
                hvals, states = update_states(barr, y_hat, marr, mu0, c, v, l, _success, _progress, pc_main, hvals, states, POW3)
            
            marr = compute_var_to_edge(marr, dv, mu0, wl); pc_main = 20     # updates the local messages to be sent to check nodes
            hvals, states = update_states(barr, y_hat, marr, mu0, c, v, l, _success, _progress, pc_main, hvals, states, POW3)

            message_arr[nbrs] = marr[ : dv].copy(); pc_main = 21            # write the updated local messages to the memory
            hvals, states = update_states(barr, y_hat, marr, mu0, c, v, l, _success, _progress, pc_main, hvals, states, POW3)

        pc_main = 22
        hvals, states = update_states(barr, y_hat, marr, mu0, c, v, l, _success, _progress, pc_main, hvals, states, POW3)

    pc_main = 23
    hvals, states = update_states(barr, y_hat, marr, mu0, c, v, l, _success, _progress, pc_main, hvals, states, POW3)

    if (0 not in y_hat) and (np.all(msg_remap(y_hat) @ H.T % 2 == 0)):  # update success flag based on whether the decoder has failed or not (not based on correct codeword)
        pc_main = 24
        hvals, states = update_states(barr, y_hat, marr, mu0, c, v, l, _success, _progress, pc_main, hvals, states, POW3)

        _success = 1; pc_main = 25
        hvals, states = update_states(barr, y_hat, marr, mu0, c, v, l, _success, _progress, pc_main, hvals, states, POW3)
    
    pc_main = 26
    hvals, states = update_states(barr, y_hat, marr, mu0, c, v, l, _success, _progress, pc_main, hvals, states, POW3)
    return hvals, states
