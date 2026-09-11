import os
import yaml
import pickle
import argparse
import subprocess
import numpy as np

from utils.readalist import get_parity_check_alist
from utils.helperfn import *

class Range:
    def __init__(self, low, high):
        self.low = low
        self.high = high

    def __call__(self, x):
        x = float(x)
        if not (self.low < x < self.high):
            raise argparse.ArgumentTypeError(
                f"value must be in ({self.low}, {self.high})"
            )
        return x
    
class ArgumentParser(argparse.ArgumentParser):
    """
        This class sets up the argument parser for the simulator with YAML config support. 
        The priority is given in the following order:
            CLI args --> YAML config args --> default args
    """
    def __init__(self, config=None, *args, **kargs,):
        ''' 
            Initialization of the class. Passes config file path 
        '''
        super().__init__(*args,**kargs)

        if config is not None:
            self.config = config
            if not os.path.exists(config):
                raise FileNotFoundError(f"Config file not found: {config}")

            try:
                with open(config, "r") as f:
                    self.parms = yaml.safe_load(f) or {}
            except yaml.YAMLError as e:
                raise ValueError(f"Invalid YAML in config file: {config}\n{e}")
        return
    
    def parse_args(self, *args, **kwargs):
        self.set_yaml_defaults()
        return super().parse_args(*args, **kwargs)

    # apply YAML defaults
    def set_yaml_defaults(self):
        """
            Set argparse defaults using YAML config.
            Should be called AFTER all add_argument calls.
        """
        defaults = {}

        for action in self._actions:
            if not action.option_strings:
                continue  # skip positional args

            key = action.dest
            matches = self._recursive_search_all(self.parms, key)

            if len(matches) > 1:
                raise ValueError(f"Ambiguous key '{key}' found multiple times in config")
            elif len(matches) == 1:
                defaults[key] = matches[0]

        self.set_defaults(**defaults)

    def _recursive_search_all(self, d, target, found=None):
        if found is None:
            found = []

        if isinstance(d, dict):
            if target in d:
                found.append(d[target])

            for v in d.values():
                self._recursive_search_all(v, target, found)

        return found

def parse_opt(cfg_file=None):
    parser = ArgumentParser(prog='mismatch cost', config=cfg_file)

    # ---------------- BASIC ARGS ----------------
    parser.add_argument("--parityfile", type=str, default="data/hamming_7x3.alist")
    parser.add_argument("--simname", type=str, default="hamming_full")

    parser.add_argument("--seed", type=int, default=100)
    parser.add_argument("--nworkers", type=int, default=None)
    parser.add_argument("--flushsze", type=int, default=500)
    parser.add_argument("--rootdir", type=str, default="results")

    parser.add_argument("--lmax", type=int, default=3)
    parser.add_argument("--nprog", type=int, default=26)
    parser.add_argument("--submatsze", type=int, default=3**11)

    parser.add_argument("--alpha_start", type=Range(0,1), default=0.1)
    parser.add_argument("--alpha_end", type=Range(0,1), default=1.0)
    parser.add_argument("--alpha_step", type=Range(0,1), default=0.1)

    parser.add_argument("--eps_start", type=Range(0,1), default=0.1)
    parser.add_argument("--eps_end", type=Range(0,1), default=1.0)
    parser.add_argument("--eps_step", type=Range(0,1), default=0.1)

    parser.add_argument("--mmc_iter", type=int, default=300)

    parser.add_argument("--optq0", type=bool, default=False)
    parser.add_argument("--opt_iter", type=int, default=500000)
  
    args = parser.parse_args()
    return args

def augment_args(args):
    rng = np.random.RandomState(args.seed)

    # --- load code ---
    n, m, dvmax, dcmax, _, _, H = get_parity_check_alist(args.parityfile)
    var2edges, check2edges, nedges = message_structure(H)

    # --- derived ---
    args.H = H
    args.n = n
    args.m = m
    args.dvmax = dvmax
    args.dcmax = dcmax
    args.dmax = max(dvmax, dcmax)

    args.nedges = nedges
    args.var2edges = var2edges
    args.check2edges = check2edges

    # --- params for main decoder/subroutines ---
    if "main" in args.simname:
        args.statelen = n + args.dmax + 7 

        # --- arrays ---
        args.alpha_arr = np.arange(args.alpha_start, args.alpha_end, args.alpha_step)
        args.eps_arr = np.arange(args.eps_start, args.eps_end, args.eps_step)

        # --- powers ---
        args.POW3 = 3 ** np.arange(n, dtype=object)
        args.POW2 = 2 ** np.arange(n, dtype=object)

    elif "check" in args.simname:
        args.statelen = 2 * args.dmax + 2 
    else:
        args.statelen = 2 * args.dmax + 4    

    args.barr = np.arange(args.statelen) + rng.uniform(size=args.statelen)

    # --- workers ---
    if args.nworkers is None:
        args.nworkers = max(1, os.cpu_count() - 1)

    args.projroot = os.getcwd()
    return args

def args_to_dict(args):
    clean = {}

    for k, v in vars(args).items():
        if isinstance(v, (int, float, str, bool)) or v is None:
            clean[k] = v
        elif hasattr(v, "tolist"):  # numpy
            clean[k] = v.tolist()
        else:
            clean[k] = str(v)  # fallback

    return clean

def save_yaml(args, path):
    yaml.dump(args_to_dict(args), open(path, "w"))

def init_directory(args):
    dirargs = dict()

    args.savedir = dirargs["savedir"] = os.path.join(args.projroot, args.rootdir, args.simname)

    dirargs['trj_dir'] = os.path.join(args.savedir, "trajectories")
    dirargs['trans_dir'] = os.path.join(args.savedir, "stoch_maps/transitions")
    dirargs['stochmat_dir'] = os.path.join(args.savedir, "stoch_maps/stochmatrix")
    dirargs['nsptrans_dir'] = os.path.join(args.savedir, "stoch_maps/nsp_transitions")
    dirargs['nspmat_dir'] = os.path.join(args.savedir, "stoch_maps/nsp_stochmatrix")
    dirargs['mmc_dir'] = os.path.join(args.savedir, "mmc_results_optq0") if args.optq0 else os.path.join(args.savedir, "mmc_results")

    createdir(**dirargs)

    PARAMS = {
                "H": args.H,
                "var2edges": args.var2edges,
                "check2edges": args.check2edges,
                "nedges": args.nedges,
                "dvmax": args.dvmax,
                "dcmax": args.dcmax,
                "lmax": args.lmax,
                "nprog": args.nprog,
                "barr": args.barr
            }

    with open(os.path.join(args.savedir, f"params_{args.simname.split('_')[0]}.pkl"), "wb") as _file:
        pickle.dump(PARAMS, _file)

    print(f"Directories are created --> parameters are saved!")
    return dirargs

# checkpoint check/create for simulation
def is_done(dirpath, stage):
    return os.path.exists(os.path.join(dirpath, f"{stage}.done"))

def mark_done(dirpath, stage):
    open(os.path.join(dirpath, f"{stage}.done"), "w").close()
