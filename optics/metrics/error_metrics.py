import numpy as np
import scipy.constants as const
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import xraylib as xrl
import os, functools
from pathlib import Path
import argparse

from ..propagators import *
from ..classes import *
from .metrics import *

script_dir = Path(__file__).resolve().parent
savedir = (script_dir / "./test_figs").resolve()

if __name__ == "__main__":
    pass