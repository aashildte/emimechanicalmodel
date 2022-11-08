"""

Åshild Telle / Simula Research Laboratory / 2022

Script for loading precomputed active tension from a npy file;
this precomputed transient is again taken from the Rice model.

"""

import os
import numpy as np
from scipy.interpolate import interp1d


def compute_active_component(time: np.ndarray):
    """

    Loads/computes interpolation of active stress, as computed for 1000 time
    steps over a single contraction using the Rice model.

    Args:
        time - numpy 1D array, time steps for single contraction
            (e.g. [0, 1, ..., 1000])

    Returns:
        np.ndarray, computed active stress over a single contraction

    """

    active_fname = os.path.join(os.path.dirname(__file__), "active.npy")
    active_ipdata = np.load(active_fname)
    time_ipdata = np.linspace(0, 1000, len(active_ipdata))

    active_fn = interp1d(time_ipdata, active_ipdata)

    return active_fn(time)
