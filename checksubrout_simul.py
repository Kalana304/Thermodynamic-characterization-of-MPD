#################################################################################
## Author       : Kalana G Abeywardena
## Created on   : Nov 2025
## Last edited  : Sept 2026
## Purpose      : runs the simulation for check subroute machine
#################################################################################

import os
import time
import shutil
import pickle
import numpy as np
from itertools import product
from multiprocessing import get_context
from concurrent.futures import ProcessPoolExecutor, as_completed

from utils.helperfn import *
from utils.argconfig import *
from utils.subroutinefn import *
from utils.mismatchfn import *

def createtrj(min_patterns, kwargs):
    """
        This fucntion calls the process_edgemsg() function to simulate check node subroutine
        and save the trajectories for a set of edge messages given.
    """
    ctx = get_context("spawn")

    t0 = time.time()
    with ProcessPoolExecutor(
                                max_workers=kwargs.nworkers,
                                mp_context=ctx,
                                initializer=worker_init,
                                initargs=(kwargs, )
                            ) as executor:

        futures  = []
        paramlist = [(im, m_, kwargs.inputsze, kwargs.flushsze, "check") for im, m_ in enumerate(min_patterns)]
        for params in paramlist:
            futures.append(executor.submit(process_edgemsg, params))

            if len(futures) >= kwargs.nworkers:
                done = futures.pop(0)
                done.result()

        for f in as_completed(futures):
            f.result()

    elapsed = (time.time() - t0) / 60
    print(f"Total time: {elapsed:.3f} min", flush=True)

def createstochmat(dirargs, kwargs):
    """
        This function uses helper functions to define state space of the check node subroutine computation simulated, 
        create the state transition matrix, and the relevant lookup tables (bijection mapping).
    """
    # listing trajectory files
    edgefiles = [
                    os.path.join(dirargs.get('trj_dir'), f)
                    for f in sorted(os.listdir(dirargs.get('trj_dir')))
                    if f.endswith(".pkl") and "_dict" not in f
                ]
    edgefiles = np.array(edgefiles)

    # creating accessed state dict and initial state dict
    _accesseddict_path = os.path.join(kwargs.savedir, 'stoch_maps/accessed.pkl')
    _initdict_path = os.path.join(kwargs.savedir, 'stoch_maps/initial.pkl')
    accessed_dict, init_dict = createstatespace(edgefiles, _accesseddict_path, _initdict_path, nworkers=kwargs.nworkers)

    ## ---------------------------- updating stochastic matrix ------------------------------ ##
    _lookuppath = os.path.join(kwargs.savedir, 'stoch_maps/lookup_tables.pkl')
    state2idx, hash2idx = createstate2idx(accessed_dict, _lookuppath)

    kwargs.sze = len(hash2idx); print(f"Size of the accesssed state space: {kwargs.sze}")
    kwargs.kappa = min(kwargs.submatsze, kwargs.sze)
    nsub_mats = np.ceil(kwargs.sze / kwargs.kappa).astype('int'); print(f"no. of submats to save = {nsub_mats}")

    # initialize the stoch matrix and trajectory dir
    initstochmatrix(
                        _sze = kwargs.sze, _nsub_mats = nsub_mats, _submat_size = kwargs.kappa, 
                        _stochpath = dirargs.get("stochmat_dir"), _transpath = dirargs.get("trans_dir")
                    )

    tic = time.time()
    with ProcessPoolExecutor(
                                max_workers=kwargs.nworkers,
                                initializer=init_worker,
                                initargs = (hash2idx, kwargs.kappa, kwargs.sze)
                            ) as ex:
        for fp in edgefiles:
            ex.submit(collect_transitions, fp, dirargs.get("trans_dir"))
    print(f"Transitions ETA: {(time.time() - tic) / 60} min")

    # updating transition matrices
    subdirs_updates = sorted(os.listdir(dirargs.get("trans_dir")))
    
    tic = time.time()
    with ProcessPoolExecutor(max_workers=kwargs.nworkers) as ex:
        for im, transfile in enumerate(subdirs_updates):
            ex.submit(update_matrix, 
                    os.path.join(dirargs.get("trans_dir"), transfile), dirargs.get("stochmat_dir"))

    print(f"Update matrix ETA: {(time.time() - tic) / 60} min")

    row_prob_sum = normalizestochmap(sze = kwargs.sze, submat_size = kwargs.kappa, _path = dirargs.get("stochmat_dir"),
                                      _rowsum = None, _normalize = False)
    row_prob_sum = normalizestochmap(sze = kwargs.sze, submat_size = kwargs.kappa, _path = dirargs.get("stochmat_dir"),
                                       _rowsum = row_prob_sum, _normalize = True)

    ## ------------------- updating stochastic matrix for non-special states --------------------- ##
    _nsplookuppath = os.path.join(kwargs.savedir, 'stoch_maps/nsp_state2idx.pkl')
    nsp_state2idx = creatensplookupdir(state_to_idx = state2idx, _nsplookuppath = _nsplookuppath, _machine = "check")

    kwargs.nspsze = len(nsp_state2idx); print(f"Size of the accesssed non-special state space: {kwargs.nspsze}")
    kwargs.nspkappa = min(kwargs.submatsze, kwargs.nspsze)
    nsub_mats_nsp = np.ceil(kwargs.nspsze / kwargs.nspkappa).astype('int'); print(f"no. of submats to save for non-special = {nsub_mats_nsp}")

    initstochmatrix(
                        _sze = kwargs.nspsze, _nsub_mats = nsub_mats_nsp, _submat_size = kwargs.nspkappa, 
                        _stochpath = dirargs.get("nspmat_dir"), _transpath = dirargs.get("nsptrans_dir")
                    )

    tic = time.time()
    with ProcessPoolExecutor(
                                max_workers=kwargs.nworkers,
                                initializer=init_worker,
                                initargs = (nsp_state2idx, kwargs.nspkappa, kwargs.nspsze)
                            ) as ex:
        for fp in edgefiles:
            ex.submit(collect_transitions_nsp, fp, dirargs.get("nsptrans_dir"), "check")

    print(f"Transitions ETA: {(time.time() - tic) / 60} min")

    # updating transition matrices
    nspsubdirs_updates = sorted(os.listdir(dirargs.get("nsptrans_dir")))

    for im, transfile in enumerate(nspsubdirs_updates):
        update_matrix(mat_trans_path = os.path.join(dirargs.get("nsptrans_dir"), transfile), matrix_path = dirargs.get("nspmat_dir"))
    
    row_prob_sum = normalizestochmap(sze = kwargs.nspsze, submat_size = kwargs.nspkappa, _path = dirargs.get("nspmat_dir"),
                                      _rowsum = None, _normalize = False)
    row_prob_sum = normalizestochmap(sze = kwargs.nspsze, submat_size = kwargs.nspkappa, _path = dirargs.get("nspmat_dir"),
                                       _rowsum = row_prob_sum, _normalize = True)

    initstate_hash = np.array(list(init_dict.keys()))
    initstate_idx = np.sort(np.array([hash2idx[v] for v in initstate_hash])) # need to sort

    return initstate_idx, nsp_state2idx
    

def runmmc(_initstate_idx, _nsp_state2idx, dirargs, kwargs):
    tic_ = time.time()
    pmmc_subr(_initidx = _initstate_idx, _nsp_state2idx = _nsp_state2idx, dirargs = dirargs, kwargs = kwargs)
    print(f"Elapsed time = {time.time() - tic_} s")

if __name__ == "__main__":
    args = parse_opt(cfg_file="data/checksubrout_config.yaml")   # set arguments
    args = augment_args(args)                                   # aux/augment arguments
    print(f"Edges={args.nedges} | dv_max = {args.dvmax} | dc_max = {args.dcmax}", flush=True)

    # initializing directories
    dirargs = init_directory(args=args)

    # defining variables needed
    min_patterns = list(product([-1, 0, 1], repeat = args.dmax)) # inputs to the check subr (all possible 3^dmax edge messages)
    args.inputsze = len(min_patterns)

    # ======================= TRAJECTORIES =======================
    if not is_done(args.savedir, "traj"):   # checkpoint look: traj creation
        print("++++++++ Running simulations :: Trajectories ++++++++")
        createtrj(min_patterns=min_patterns, kwargs=args)
        mark_done(args.savedir, "traj")
    else:
        print("Skipping trajectories (already created)!")

    # ======================= STOCH MATRIX =======================
    if not is_done(args.savedir, "stoch"):  # checkpoint look: stoch mat creation
        print("++++++++ Running simulations :: Stoch matrix ++++++++")
        initstate_idx, nsp_state2idx = createstochmat(dirargs = dirargs, kwargs = args)
        mark_done(args.savedir, "stoch")
    else:
        print("Skipping stoch matrix (already created)!")
        nsp_state2idx = pickle.load(open(os.path.join(args.savedir, 'stoch_maps/nsp_state2idx.pkl'), "rb"))

        _, hash2idx = pickle.load(open(os.path.join(args.savedir, 'stoch_maps/lookup_tables.pkl'), "rb")) 
        init_dict = pickle.load(open(os.path.join(args.savedir, 'stoch_maps/initial.pkl'), "rb"))

        initstate_hash = np.array(list(init_dict.keys()))
        initstate_idx = np.sort(np.array([hash2idx[v] for v in initstate_hash])) # need to sort

        args.sze = len(hash2idx); args.kappa = min(args.submatsze, args.sze)
        args.nspsze = len(nsp_state2idx); args.nspkappa = min(args.submatsze, args.nspsze)

        del init_dict, hash2idx, initstate_hash

    shutil.rmtree(dirargs.get("trans_dir"))
    shutil.rmtree(dirargs.get("nsptrans_dir"))
    shutil.rmtree(dirargs.get("trj_dir"))

    # ======================= MMC =======================
    if not is_done(args.savedir, "mmc"):    # checkpoint look: mmc valuation
        print("++++++++ Running simulations mmc ++++++++")
        runmmc(initstate_idx, nsp_state2idx, dirargs, kwargs=args)

        mark_done(args.savedir, "mmc")
    else:
        print("Skipping mmc (already done)")

    if args.optq0:
        if not is_done(args.savedir, "mmc_optq0"):    # checkpoint look: mmc_optq0 valuation
            print("++++++++ Running simulations mmc for optq0 ++++++++")
            runmmc(initstate_idx, nsp_state2idx, dirargs, kwargs=args)

            mark_done(args.savedir, "mmc_optq0")
        else:
            print("Skipping mmc for optq0 (already done)")

    save_yaml(args, os.path.join(args.savedir, f"{args.simname}_updated.yaml"))
    print("++++++++++++++++++++ Finished simulations +++++++++++++++++++++++")