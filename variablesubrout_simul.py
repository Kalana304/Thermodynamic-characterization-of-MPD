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
    ctx = get_context("spawn")

    t0 = time.time()
    with ProcessPoolExecutor(
                                max_workers=kwargs.nworkers,
                                mp_context=ctx,
                                initializer=worker_init,
                                initargs=(kwargs, )
                            ) as executor:

        futures  = []
        paramlist = [(im, m_, kwargs.inputsze, "variable") for im, m_ in enumerate(min_patterns)]
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
    pass

def runmmc(_initstate_idx, _nsp_state2idx, _codewords, dirargs, kwargs):
    pass

if __name__ == "__main__":
    args = parse_opt(cfg_file="data/variablesubrout_config.yaml")   # set arguments
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