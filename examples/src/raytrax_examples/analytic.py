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


def isothermal_slab(absorptivity: float, length: float, edges: np.ndarray) -> np.ndarray:
    """Give the mean incident radiation over each bin of an infinite hot slab between two cold black walls.

    Parameters
    ----------
    absorptivity
        The absorption coefficient of the gas.
    length
        The distance between the walls, the first of which is at zero.
    edges
        The bin edges along the depth of the slab.

    Returns
    -------
    incident
        The incident radiation per unit emissive power of the gas, one value per bin.
    """
    near, far = absorptivity * edges[:-1], absorptivity * edges[1:]
    depth = absorptivity * length
    walls = expn(3, near) - expn(3, far) + expn(3, depth - far) - expn(3, depth - near)
    return 4 - 2 * walls / (far - near)


def isothermal_slab_point(absorptivity: float, length: float, position: np.ndarray) -> np.ndarray:
    """Give the incident radiation at points of an infinite hot slab between two cold black walls.

    Parameters
    ----------
    absorptivity
        The absorption coefficient of the gas.
    length
        The distance between the walls, the first of which is at zero.
    position
        The positions along the depth of the slab.

    Returns
    -------
    incident
        The incident radiation per unit emissive power of the gas at each position.
    """
    return 4 - 2 * (expn(2, absorptivity * position) + expn(2, absorptivity * (length - position)))
