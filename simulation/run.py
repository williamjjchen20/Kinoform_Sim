import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import scipy.constants as const
import xraylib
import functools, sys, os
import argparse

from .setup import *

parser = argparse.ArgumentParser(prog="Optics Simulation", description="")
parser.add_argument('N', type=int, nargs='?', default=1024, help="Numerical resolution")
parser.add_argument('Lx', type=float, nargs='')
parser.add_argument('Lz', type=float, required=True)
parser.add_argument('Ly', nargs="?", default=None)

if __name__ == "__main__":
    args = parser.parse_args()
    Lx = args.Lx

