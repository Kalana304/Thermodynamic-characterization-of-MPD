import os
import numpy as np
from itertools import product

from utils.helperfn import ptrn2outstr

## Global variabels
GLOB_DMAX = None
GLOB_BARR = None
GLOB_SAVEDIR = None

MOUT_STATES = None
MU0_STATES = None
WL_STATES = None

def worker_init(args):
    global GLOB_DMAX, GLOB_BARR, GLOB_SAVEDIR
    global MOUT_STATES, MU0_STATES, WL_STATES

    # prevent BLAS thread explosion
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["NUMEXPR_NUM_THREADS"] = "1"

    GLOB_BARR = args.barr
    GLOB_SAVEDIR = args.savedir
    MOUT_STATES = np.array(list(product([-1, 0, 1], repeat=args.dmax)))

    if "variable" in args.simname:
        GLOB_DMAX = args.dvmax
        WL_STATES = [1, 2]
        MU0_STATES = np.array(list(product([-1, 0, 1], repeat=1)))
    elif "check" in args.simname:
        GLOB_DMAX = args.dcmax
    else:
        print(f"{args.simname} doesn't contain var or check!")
        raise

def process_edgemsg(args):
    """
        Process a possible edge message over Tanner graph. Since the soubroutine
        is designed to take any string of symbols and update them based on 
        message update rule, we consider all 3^{dmax} inputs possible.
    """
    im, m_, inputsze, mode_ = args
    try:
        min = np.array(m_)
        hasharray = []

        # Save file name and path
        filename = f"statehash_min_{ptrn2outstr(min)}.pkl"
        filepath = os.path.join(GLOB_SAVEDIR, "trajectories", filename)
        
        for mout in MOUT_STATES:
            mout = np.array(mout)
            if mode_ == "variable": 
                for mu0 in MU0_STATES:
                    for wl in WL_STATES:
                        hvals = var2edgesubr(
                                                barr=GLOB_BARR, m_in=min, m_out=mout, 
                                                mu0v=mu0, wl0=wl, dvMax=GLOB_DMAX
                                            )
                        hasharray.append(hvals)
            elif mode_ == "check":
                hvals = check2edgesubr(barr=GLOB_BARR, m_in=min, m_out=mout, dcMax=GLOB_DMAX)
                hasharray.append(hvals)

        hasharray = np.array(hasharray)
        # saving hashvalues, unique set of hashvalues of accessed states, and hashes of initial states
        np.savez_compressed(filepath, hash = hasharray, accessed = np.unique(hasharray), init = hasharray[:, 0])
    
        print(f"Completed edge msg {im + 1} / {inputsze}", flush=True)

    except Exception as e:
        print(f"[Worker crash] im = {im} | {e}", flush=True)
        raise

def update_states(barr, m_in, m_out, mu0, wl0, subridx, pc_, hvals_, subroute='check'):
    dn = len(m_in)        # length of incoming message reg

    if subroute == "check":
        hvals_.append( np.dot(m_in, barr[ : dn]) + np.dot(m_out, barr[dn : 2 * dn]) + 
                        subridx * barr[2 * dn] + pc_ * barr[2 * dn + 1]
                    )
    elif subroute == "variable":
        hvals_.append( np.dot(m_in, barr[ : dn]) + np.dot(m_out, barr[dn : 2 * dn]) + 
                        mu0 * barr[2 * dn] + wl0 * barr[2 * dn + 1] + 
                        subridx * barr[2 * dn + 2] + pc_ * barr[2 * dn + 3]
                    )      
    else:
        print(f"{subroute} is not a valid argument!")
    return hvals_

def check2edgesubr(barr, m_in, m_out, dcMax):
    """ 
        updates the messages to be sent by a check node.
        state space: (msgin, msgout, v, pc_c): 3^4 x 3^4 x 4 x 4
    """
    msgin = m_in.copy()         # initialize input reg with edge messages
    msgout = m_out.copy()       # initialize output reg uniformely
    v = 0                       # loop counter for neighbour var node
    pc_check = 0                # program counter for check subroutine

    hvals = update_states(barr, msgin, msgout, None, None, v, pc_check, [], subroute='check')

    for v in range(1, dcMax + 1):
        pc_check = 1
        hvals = update_states(barr, msgin, msgout, None, None, v, pc_check, hvals, subroute='check')

        # For this simulation, we assume below operation is performed by a separate multiplier (subsystem) 
        # for which the mismatch cost is ignored. Thus, we get a looser mmc estimation.  
        msgout[v - 1] = np.prod(np.delete(msgin[ : dcMax], v - 1))      
        pc_check = 2
        hvals = update_states(barr, msgin, msgout, None, None, v, pc_check, hvals, subroute='check')

    pc_check = 3
    hvals = update_states(barr, msgin, msgout, None, None, v, pc_check, hvals, subroute='check')

    return hvals

def var2edgesubr(barr, m_in, m_out, mu0v, wl0, dvMax):
    """ 
        updates the messages to be sent by a var node.
        state space: (msgin, msgout, mu0v, wl0, c, pc_v): 3^4 x 3^4 x 4 x 4
    """
    msgin = m_in.copy()         # initialize input reg with edge messages
    muobs = mu0v                # intiialize input reg with obs symbols from channel output {-1, 0, 1}
    wlin = wl0                  # for gallager, the obs weight is an input from main decoder
    msgout = m_out.copy()       # initialize output reg uniformely
                   
    c = 0                       # loop counter for neighbour var node
    pc_var = 0                  # program counter for check subroutine

    hvals = update_states(barr, msgin, msgout, muobs, wlin, c, pc_var, [], subroute='variable')

    for c in range(1, dvMax + 1):
        pc_var = 1
        hvals = update_states(barr, msgin, msgout, muobs, wlin, c, pc_var, hvals, subroute='variable')

        # For this simulation, we assume below operation is performed by a separate adder/comparator (subsystem)
        # for which the mismatch cost is ignored. Thus, we get a looser mmc estimation.  
        msgout[c - 1] = np.sign(wlin * muobs + np.sum(np.delete(msgin[ : dvMax], c - 1)))      
        pc_var = 2
        hvals = update_states(barr, msgin, msgout, muobs, wlin, c, pc_var, hvals, subroute='variable')
        

    pc_var = 3
    hvals = update_states(barr, msgin, msgout, muobs, wlin, c, pc_var, hvals, subroute='variable')
    
    return hvals