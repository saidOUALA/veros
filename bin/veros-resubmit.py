#!/usr/bin/env python

"""Performs several runs of Veros back to back, using the previous run as restart input.

Intended to be used with scheduling systems (e.g. SLURM or PBS).
"""

import argparse
import subprocess

def parse_cli():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("N_SUBMITS", type=int, help="total number of resubmits")
    parser.add_argument("LENGTH_PER_RUN", type=float, help="length (in seconds) of each run")
    parser.add_argument("VEROS_CMD", help="the command that is used to call veros")
    return parser.parse_args()

def call_veros(cmd, n, runlen):
    args = "-s restart_output_filename \"{{identifier}}_restart_{n}.h5\" -s runlen {runlen}".format(n=n, runlen=runlen)
    if n:
        args += "-s restart_input_filename \"{{identifier}}_restart_{n_prev}.h5\"".format(n_prev=n-1)
    print(" > " + cmd + args)
    subprocess.check_call(cmd + args, shell=True)

if __name__ == "__main__":
    args = parse_cli()
    for n in range(args.N_SUBMITS):
        call_veros(args.VEROS_CMD, n, args.LENGTH_PER_RUN)