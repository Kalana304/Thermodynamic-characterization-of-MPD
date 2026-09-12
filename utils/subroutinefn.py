#################################################################################
## Author       : Kalana G Abeywardena
## Created on   : Nov 2025
## Last edited  : Sept 2026
## Purpose      : defines computations carried in each subroutine to generate 
##                trajectroy files for each possible edge message they would get
#################################################################################

import os
import gc
import pickle
import numpy as np
from itertools import product

from utils.helperfn import ptrn2outstr

## Global variabels (so workers have access to them easily than passing them as arguments)
GLOB_DMAX = None
GLOB_BARR = None
GLOB_SAVEDIR = None

MOUT_STATES = None
MU0_STATES = None
WL_STATES = None

POW3 = None

def worker_init(args):
    """
        This function initializes the common parameters that will be utilized by
        parallely running workers during the simulation.
    """
    global GLOB_DMAX, GLOB_BARR, GLOB_SAVEDIR
    global MOUT_STATES, MU0_STATES, WL_STATES, POW3

    # prevent BLAS thread explosion
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["NUMEXPR_NUM_THREADS"] = "1"

    GLOB_BARR = args.barr
    GLOB_SAVEDIR = args.savedir
    MOUT_STATES = np.array(list(product([-1, 0, 1], repeat=args.dmax)))
    POW3 = args.POW3

    if "variable" in args.simname:
        # some global variables that are specific for variable subroutine
        GLOB_DMAX = args.dvmax
        WL_STATES = [1, 2]
        MU0_STATES = [-1, 0, 1]
    elif "check" in args.simname:
        GLOB_DMAX = args.dcmax
    else:
        print(f"{args.simname} doesn't contain var or check!")
        raise

def process_edgemsg(args):
    """
        Process a possible edge message over the given Tanner graph. Since the soubroutine
        is designed to take any string of symbols and update them based on message update rule
        specificied for each subroutine, we consider all 3^{dmax} inputs possible.
    """
    im, m_, inputsze, flush_sze, mode_ = args
    try:
        min = np.array(m_)

        accessed_dict = dict()
        init_dict = dict()

        # save file name and path
        filename = f"statehash_medge_{ptrn2outstr(min)}.pkl"
        filepath = os.path.join(GLOB_SAVEDIR, "trajectories", filename)

        buffer = []

        # open trajectory file
        traj_file = open(filepath, "ab")

        for mout in MOUT_STATES:
            mout = np.array(mout)
            if mode_ == "variable": 
                # the following two state variables are only applicable to var subroutines
                for mu0 in MU0_STATES:
                    for wl in WL_STATES:
                        hvals, states = var2edgesubr(
                                                barr=GLOB_BARR, m_in=min, m_out=mout, 
                                                mu0v=mu0, wl0=wl, dvMax=GLOB_DMAX,
                                                POW3=POW3
                                            )
                        buffer.append((hvals, states[0], states[-1]))
                        
                        for h, s in zip(hvals, states):
                            accessed_dict[h] = s

                        init_dict[hvals[0]] = states[0]

                        # flush periodically
                        if len(buffer) >= flush_sze:
                            pickle.dump(buffer, traj_file, protocol=pickle.HIGHEST_PROTOCOL)
                            buffer.clear()

            elif mode_ == "check":
                hvals, states = check2edgesubr(
                                                barr=GLOB_BARR, m_in=min, m_out=mout, 
                                                dcMax=GLOB_DMAX, POW3=POW3
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
   
        print(f"Completed edge msg {im + 1} / {inputsze}", flush=True)

        del accessed_dict, init_dict
        gc.collect()

    except Exception as e:
        print(f"[Worker crash] im = {im} | {e}", flush=True)
        raise

def update_states(barr, m_in, m_out, mu0, wl0, subridx, pc_, hvals_, states_, POW3, subroute='check'):
    """
        Function to update the states and their hash values for each clock cycle.
        
        args:
            barr (numpy.array)      : random values used to hash the state values
            m_in (numpy.array)      : input edge messages to the subroutine
            m_out (numpy.array)     : output edge messages from the subroutine
            mu0 (int)               : channel observation (only for `variable` subroutine)
            wl0 (int)               : channel observation weightage (only for `variable` subroutine, and gallager algo)
            subridx (int)           : loop counter for subroutine
            pc_ (int)               : program counter for the subroutine
            hvals_ (list)           : a list to which hashvalues are appeneded to keep track of trajectories
            states_ (list)          : a list to which the state values are appended
            subroute (str)          : which subroutine is simulated
        
        return:
            hvals_ (list)           : updated list of hash values
            states_ (list)          : updated list of state values
    """
    dn = len(m_in)        # length of incoming message reg

    if subroute == "check":
        hvals_.append( 
                        np.dot(m_in, barr[ : dn]) + np.dot(m_out, barr[dn : 2 * dn]) + 
                        subridx * barr[2 * dn] + pc_ * barr[2 * dn + 1]
                    )
        states_.append((np.sum(np.dot(POW3, m_in), dtype=int), np.sum(np.dot(POW3, m_out), dtype=int), subridx, pc_))

    elif subroute == "variable":
        hvals_.append( np.dot(m_in, barr[ : dn]) + mu0 * barr[dn] + 
                       wl0 * barr[dn + 1] + np.dot(m_out, barr[dn + 2 : 2 * dn + 2]) + 
                       subridx * barr[2 * dn + 2] + pc_ * barr[2 * dn + 3]
                    )      
        states_.append((np.sum(np.dot(POW3, m_in), dtype=int), int(mu0), int(wl0), np.sum(np.dot(POW3, m_out), dtype=int), subridx, pc_))

    else:
        print(f"{subroute} is not a valid argument!")

    return hvals_, states_

def check2edgesubr(barr, m_in, m_out, dcMax, POW3):
    """ 
        This function carries out the message updates to be sent by a check node based on the edge messages
        it received by some neighbourhood. The message update rule is common to all message passing algorithms
        where we take the product of messages excluding the message of the var node to which the outgoing message
        is intended for. 

        state space: (msgin, msgout, v, pc_c): 3^4 x 3^4 x 4 x 4

        args:
            barr (numpy.array)      : random values used to hash the state values
            m_in (numpy.array)      : input edge messages to subroutine
            m_out (numpy.array)     : random initialization of internal register
            dcMax (int)             : degree of each check node
            POW3 (numpy.array)      : power of 3 used to encode state
        
        return:
            hvals (list)            : list of hash values of the visited states along the computation
            states (list)           : list of state values of the visited states along the computation
    """
    msgin = m_in.copy()         # initialize input reg with edge messages
    msgout = m_out.copy()       # initialize output reg uniformely
    v = 0                       # loop counter for neighbour var node
    pc_check = 0                # program counter for check subroutine

    hvals, states = update_states(barr, msgin, msgout, None, None, v, pc_check, [], [], POW3, subroute='check')

    for v in range(1, dcMax + 1):
        pc_check = 1
        hvals, states = update_states(barr, msgin, msgout, None, None, v, pc_check, hvals, states, POW3, subroute='check')

        # For this simulation, we assume below operation is performed by a separate multiplier (subsystem) 
        # for which the mismatch cost is ignored. Thus, we get a looser mmc estimation.  
        msgout[v - 1] = np.prod(np.delete(msgin[ : dcMax], v - 1))      
        pc_check = 2
        hvals, states = update_states(barr, msgin, msgout, None, None, v, pc_check, hvals, states, POW3, subroute='check')

    pc_check = 3
    hvals, states = update_states(barr, msgin, msgout, None, None, v, pc_check, hvals, states, POW3, subroute='check')

    return hvals, states

def var2edgesubr(barr, m_in, m_out, mu0v, wl0, dvMax, POW3):
    """ 
        This function carries out the message updates to be sent by a variable node based on the edge messages
        it received by some neighbourhood. The message update rule used here is based on Gallager A algorithm;
        apart from the relative weighting on the channel observation (mu0) by wl0, it follows any other message
        passing algoroithms, where we sum incoming messages excpet of the check node for which we compute the 
        outgoing messsage, with the channel observation. 

        state space: (msgin, muobs, wlin, msgout, c, pc_var): 3^4 x 3 x 2 x 3^4 x 3 x 4

        args:
            barr (numpy.array)      : random values used to hash the state values
            m_in (numpy.array)      : input edge messages to subroutine
            m_out (numpy.array)     : random initialization of internal register
            mu0v (int)              : channel observation
            wl0 (int)               : relative weighting of channel observations (for most algoerithms, this is a const. 1)
            dvMax (int)             : degree of each check node
            POW3 (numpy.array)      : power of 3 used to encode state
        
        return:
            hvals (list)            : list of hash values of the visited states along the computation
            states (list)           : list of state values of the visited states along the computation
    """
    msgin = m_in.copy()         # initialize input reg with edge messages (input)
    muobs = mu0v                # intiialize input reg with obs symbols from channel output {-1, 0, 1} (input)
    wlin = wl0                  # for gallager, the obs weight is an input from main decoder (input)
    msgout = m_out.copy()       # initialize output reg uniformely (output)
                   
    c = 0                       # loop counter for neighbour var node
    pc_var = 0                  # program counter for check subroutine

    hvals, states = update_states(barr, msgin, msgout, muobs, wlin, c, pc_var, [], [], POW3, subroute='variable')

    for c in range(1, dvMax + 1):
        pc_var = 1
        hvals, states = update_states(barr, msgin, msgout, muobs, wlin, c, pc_var, hvals, states, POW3, subroute='variable')

        # For this simulation, we assume below operation is performed by a separate adder/comparator (subsystem)
        # for which the mismatch cost is ignored. Thus, we get a looser mmc estimation.  
        msgout[c - 1] = np.sign(wlin * muobs + np.sum(np.delete(msgin[ : dvMax], c - 1)))      
        pc_var = 2
        hvals, states = update_states(barr, msgin, msgout, muobs, wlin, c, pc_var, hvals, states, POW3, subroute='variable')
        
    pc_var = 3
    hvals, states = update_states(barr, msgin, msgout, muobs, wlin, c, pc_var, hvals, states, POW3, subroute='variable')
    
    return hvals, states