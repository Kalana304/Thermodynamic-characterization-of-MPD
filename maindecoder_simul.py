import os
import gc
import time
import shutil
import pickle
import numpy as np
from itertools import product
from multiprocessing import get_context
from concurrent.futures import ProcessPoolExecutor, as_completed

from utils.helperfn import *
from utils.argconfig import *
from utils.decoderfn import *
from utils.mismatchfn import *
from utils.readalist import get_parity_check_alist

def createtrj(y_patterns, kwargs):
    ctx = get_context("spawn")

    t0 = time.time()
    with ProcessPoolExecutor(
                                max_workers=kwargs.nworkers,
                                mp_context=ctx,
                                initializer=worker_init,
                                initargs=(kwargs, )
                            ) as executor:

        futures  = []
        paramlist = [(iy, y_, kwargs.inputsze, kwargs.flushsze) for iy, y_ in enumerate(y_patterns)]
        for params in paramlist:
            futures.append(executor.submit(process_pattern, params))

            if len(futures) >= kwargs.nworkers:
                done = futures.pop(0)
                done.result()

        for f in as_completed(futures):
            f.result()

    elapsed = (time.time() - t0) / 60
    print(f"Total time: {elapsed:.3f} min", flush=True)

def createstochmat(dirargs, kwargs):
    # listing trajectory files
    errfiles = [
                    os.path.join(dirargs.get('trj_dir'), f)
                    for f in sorted(os.listdir(dirargs.get('trj_dir')))
                    if f.endswith(".pkl") and "_dict" not in f
                ]
    errfiles = np.array(errfiles)

    # creating accessed state dict and initial state dict
    _accesseddict_path = os.path.join(kwargs.savedir, 'stoch_maps/accessed.pkl')
    _initdict_path = os.path.join(kwargs.savedir, 'stoch_maps/initial.pkl')
    accessed_dict, init_dict = createstatespace(errfiles, _accesseddict_path, _initdict_path, nworkers=kwargs.nworkers)

    ## ---------------------------- updating stochastic matrix ------------------------------ ##
    _lookuppath = os.path.join(kwargs.savedir, 'stoch_maps/lookup_tables.pkl')
    state2idx, hash2idx = createstate2idx(accessed_dict, _lookuppath)

    kwargs.sze = len(hash2idx); print(f"Size of the accesssed state space: {kwargs.sze}")
    kwargs.kappa = min(kwargs.submatsze, kwargs.sze)
    nsub_mats = np.ceil(kwargs.sze / kwargs.kappa).astype('int'); print(f"no. of submats to save = {nsub_mats}")

    # initialize the stoch matrix and trajectory dir
    initstochmatrix(kwargs.sze, nsub_mats, kwargs.kappa, dirargs.get("stochmat_dir"), dirargs.get("trans_dir"))

    tic = time.time()
    with ProcessPoolExecutor(
                                max_workers=kwargs.nworkers,
                                initializer=init_worker,
                                initargs = (hash2idx, kwargs.kappa, kwargs.sze)
                            ) as ex:
        for fp in errfiles:
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

    shutil.rmtree(dirargs.get("trans_dir"))

    row_prob_sum = normalizestochmap(kwargs.sze, kwargs.kappa, _path = dirargs.get("stochmat_dir"),
                                      _rowsum = None, _normalize = False)
    
    row_prob_sum = normalizestochmap(kwargs.sze, kwargs.kappa, _path = dirargs.get("stochmat_dir"),
                                      _rowsum = row_prob_sum, _normalize = True)

    ## ------------------- updating stochastic matrix for non-special states --------------------- ##
    _nsplookuppath = os.path.join(kwargs.savedir, 'stoch_maps/nsp_state2idx.pkl')
    nsp_state2idx = creatensplookupdir(state2idx, _nsplookuppath)

    kwargs.nspsze = len(nsp_state2idx); print(f"Size of the accesssed non-special state space: {kwargs.nspsze}")
    kwargs.nspkappa = min(kwargs.submatsze, kwargs.nspsze)
    nsub_mats_nsp = np.ceil(kwargs.nspsze / kwargs.nspkappa).astype('int'); print(f"no. of submats to save for non-special = {nsub_mats_nsp}")

    initstochmatrix(kwargs.nspsze, nsub_mats_nsp, kwargs.nspkappa, dirargs.get("nspmat_dir"), dirargs.get("nsptrans_dir"))

    tic = time.time()
    with ProcessPoolExecutor(
                                max_workers=kwargs.nworkers,
                                initializer=init_worker,
                                initargs = (nsp_state2idx, kwargs.nspkappa, kwargs.nspsze)
                            ) as ex:
        for fp in errfiles:
            ex.submit(collect_transitions_nsp, fp, dirargs.get("nsptrans_dir"))

    print(f"Transitions ETA: {(time.time() - tic) / 60} min")

    # updating transition matrices
    nspsubdirs_updates = sorted(os.listdir(dirargs.get("nsptrans_dir")))

    for im, transfile in enumerate(nspsubdirs_updates):
        update_matrix(os.path.join(dirargs.get("nsptrans_dir"), transfile), dirargs.get("nspmat_dir"))
    
    shutil.rmtree(dirargs.get("nsptrans_dir"))

    row_prob_sum = normalizestochmap(kwargs.nspsze, kwargs.nspkappa, _path = dirargs.get("nspmat_dir"),
                                      _rowsum = None, _normalize = False)
    row_prob_sum = normalizestochmap(kwargs.nspsze, kwargs.nspkappa, _path = dirargs.get("nspmat_dir"),
                                       _rowsum = row_prob_sum, _normalize = True)

    initstate_hash = np.array(list(init_dict.keys()))
    initstate_idx = np.sort(np.array([hash2idx[v] for v in initstate_hash])) # need to sort

    shutil.rmtree(dirargs.get("trj_dir"))

    return initstate_idx, nsp_state2idx

def runmmc(_initstate_idx, _nsp_state2idx, _codewords, dirargs, kwargs):
    tic_ = time.time()
    with ProcessPoolExecutor(max_workers=kwargs.nworkers) as ex:
        for alpha in kwargs.alpha_arr:
            for eps in kwargs.eps_arr:
                ex.submit(pmmc, alpha, eps, _initstate_idx, _nsp_state2idx, _codewords, dirargs, kwargs)

    elapsed_time = (time.time() - tic_) / 3600
    print(f"Elapsed time = {elapsed_time} hrs --> {elapsed_time / (len(kwargs.alpha_arr) * len(kwargs.eps_arr))} hrs per sim")

if __name__ == "__main__":
    args = parse_opt(cfg_file="data/maindecoder_config.yaml")   # set arguments
    args = augment_args(args)                                   # aux/augment arguments
    
    print(f"Edges={args.nedges} | dv_max = {args.dvmax} | dc_max = {args.dcmax}", flush=True)

    # initializing directories
    dirargs = init_directory(args=args)

    # defining variables needed
    y_patterns = list(product([-1, 0, 1], repeat = args.n)) # inputs to the decoder (all possible 3^n erasure patterns)
    args.inputsze = len(y_patterns)

    I = np.eye(4)
    G = np.append(args.H[:, 3 :].T, I, axis = 1); # corresponding generator matrix

    messages = [(((i & (1 << np.arange(args.n - args.m)))) > 0).astype(int) for i in range(2**(args.n - args.m))]
    codewords_bin = np.array([m @ G % 2 for m in messages])     # corresponding codewords C \subset {0, 1}^n
    codewords_gall = codewords_bin.copy()
    codewords_gall[codewords_gall == 0] = -1                    # mapping codewords to {-1, 1}^n space

    codeword_ = np.array([np.dot(args.POW3, x) for x in codewords_gall], dtype = int)   # convert {-1, 1}^n to decimal values
    codeword_idx = np.argsort(codeword_)                # sort the codewords based on their decimal rep. (as state space is lex sorted)
    codewords_bin_sort = codewords_bin[codeword_idx]    # sort the binary versions of codewords in the same order 

    ValidCodewords = np.array([np.dot(args.POW2, m) for m in codewords_bin_sort], dtype = int)

    # ======================= TRAJECTORIES =======================
    if not is_done(args.savedir, "traj"):   # checkpoint look: traj creation
        print("++++++++ Running simulations :: Trajectories ++++++++")
        createtrj(y_patterns=y_patterns, kwargs=args)
        mark_done(args.savedir, "traj")
    else:
        print("Skipping trajectories (already created)!")

    # ======================= STOCH MATRIX =======================
    if not is_done(args.savedir, "stoch"):  # checkpoint look: stoch mat creation
        print("++++++++ Running simulations :: Stoch matrix ++++++++")
        initstate_idx, nsp_state2idx = createstochmat(dirargs, kwargs=args)
        mark_done(args.savedir, "stoch")
    else:
        print("Skipping stoch matrix (already created)!")
        nsp_state2idx = pickle.load(open(os.path.join(args.savedir, 'stoch_maps/nsp_state2idx.pkl'), "rb"))

        _, hash2idx = pickle.load(open(os.path.join(args.savedir, 'stoch_maps/lookup_tables.pkl'), "rb")) 
        init_dict = pickle.load(open(os.path.join(args.savedir, 'stoch_maps/initial.npz'), "rb"))

        initstate_hash = np.array(list(init_dict.keys()))
        initstate_idx = np.sort(np.array([hash2idx[v] for v in initstate_hash])) # need to sort

        args.sze = len(hash2idx); args.kappa = min(args.submatsze, args.sze)
        args.nspsze = len(nsp_state2idx); args.nspkappa = min(args.submatsze, args.nspsze)

        del init_dict, hash2idx, initstate_hash

    # ======================= MMC =======================
    if not is_done(args.savedir, "mmc"):    # checkpoint look: mmc valuation
        print("++++++++ Running simulations mmc ++++++++")
        runmmc(initstate_idx, nsp_state2idx, ValidCodewords, dirargs, kwargs=args)

        mark_done(args.savedir, "mmc")
    else:
        print("Skipping mmc (already done)")

    if args.optq0 and  (not is_done(args.savedir, "mmc_optq0")):    # checkpoint look: mmc_optq0 valuation
        print("++++++++ Running simulations mmc for optq0 ++++++++")
        runmmc(initstate_idx, nsp_state2idx, ValidCodewords, dirargs, kwargs=args)

        mark_done(args.savedir, "mmc_optq0")
    else:
        print("Skipping mmc for optq0 (already done)")

    save_yaml(args, os.path.join(args.savedir, f"{args.simname}_updated.yaml"))
    print("++++++++++++++++++++ Finished simulations +++++++++++++++++++++++")
    
    



