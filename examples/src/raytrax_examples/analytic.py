"""Closed-form solutions the verification examples compare against."""

import numpy as np
from scipy.special import expn


def tangent_slab(absorptivity: float, origin: float, edges: np.ndarray) -> np.ndarray:
    """Give the mean volumetric heating over each bin of an infinite cold slab with one black wall.

    Parameters
    ----------
    absorptivity
        The absorption coefficient of the gas.
    origin
        The position of the hot wall.
    edges
        The bin edges along the depth of the slab.

    Returns
    -------
    heating
        The heating per unit emissive power, one value per bin.
    """
    near, far = absorptivity * (edges[:-1] - origin), absorptivity * (edges[1:] - origin)
    return 2 * (expn(3, near) - expn(3, far)) / (edges[1:] - edges[:-1])


def tangent_slab_point(absorptivity: float, origin: float, position: np.ndarray) -> np.ndarray:
    """Give the volumetric heating at points of an infinite cold slab with one black wall.

    Parameters
    ----------
    absorptivity
        The absorption coefficient of the gas.
    origin
        The position of the hot wall.
    position
        The positions along the depth of the slab.

    Returns
    -------
    heating
        The heating per unit emissive power at each position.
    """
    return 2 * absorptivity * expn(2, absorptivity * (position - origin))
