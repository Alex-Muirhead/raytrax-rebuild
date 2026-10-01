"""Binned heating profiles, and their error against the tangent-slab solution."""

import itertools
from typing import NamedTuple

import numpy as np

from raytrax.paths import RADIAL
from raytrax_examples.analytic import tangent_slab, tangent_slab_point
from raytrax_examples.cases import Case


class Band(NamedTuple):
    """The radial profile over one axial band.

    Attributes
    ----------
    lo
        The low-x edge of the band.
    hi
        The high-x edge of the band.
    heating
        The mean heating in each radial shell.
    exact
        The tangent-slab mean over the band, uniform in radius.
    """

    lo: float
    hi: float
    heating: np.ndarray
    exact: float


def rebin(position: np.ndarray, values: np.ndarray, weight: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """Give the weighted mean of cell values over each bin.

    Parameters
    ----------
    position
        The coordinate of each cell that selects its bin.
    values
        The value of each cell.
    weight
        The measure of each cell.
    edges
        The bin edges.

    Returns
    -------
    means
        The mean of each bin, NaN where a bin holds no cell.
    """
    index = np.clip(np.digitize(position, edges) - 1, 0, edges.size - 2)
    total = np.bincount(index, values * weight, edges.size - 1)
    measure = np.bincount(index, weight, edges.size - 1)
    return np.divide(total, measure, out=np.full(edges.size - 1, np.nan), where=measure > 0)


def axial_profile(
    case: Case, heating: np.ndarray, kappa: float, bins: int = 26
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Give the Monte-Carlo and tangent-slab heating against depth.

    Parameters
    ----------
    case
        The case that was traced.
    heating
        The volumetric heating of each cell.
    kappa
        The absorption coefficient of the gas.
    bins
        The number of bin edges.

    Returns
    -------
    edges
        The bin edges along x.
    profile
        The Monte-Carlo mean over each bin.
    exact
        The tangent-slab mean over each bin.
    """
    centroid, vertices = case.mesh.geometry.cells.centroid, case.mesh.geometry.vertices
    lo, hi = vertices[:, 0].min(), vertices[:, 0].max()
    edges = np.linspace(lo, hi, bins)
    profile = rebin(centroid[:, 0], heating, case.geom.cells.volume, edges)
    return edges, profile, tangent_slab(kappa, lo, edges)


def radial_profile(
    case: Case, heating: np.ndarray, kappa: float, bands: int = 5
) -> tuple[np.ndarray, list[Band]]:
    """Give the heating against radius in a few axial bands.

    Parameters
    ----------
    case
        The case that was traced.
    heating
        The volumetric heating of each cell.
    kappa
        The absorption coefficient of the gas.
    bands
        The number of band edges along x.

    Returns
    -------
    shells
        The shell edges along the radius.
    bands
        One `Band` per axial band.

    Raises
    ------
    ValueError
        If the case has no axis: it is a planar 2D case.
    """
    mesh = case.mesh
    if mesh.topology.dim != 3 and not case.axisymmetric:
        raise ValueError("a planar 2D case has no axis to measure a radius from")

    centroid, vertices = mesh.geometry.cells.centroid, mesh.geometry.vertices
    # Revolved, the radius is stored outright; in 3D it has to be measured.
    if case.axisymmetric:
        radius, edge = centroid[:, RADIAL], vertices[:, RADIAL]
    else:
        radius, edge = np.linalg.norm(centroid[:, 1:], axis=-1), np.linalg.norm(vertices[:, 1:], axis=-1)

    lo = vertices[:, 0].min()
    band_edges = np.linspace(lo, vertices[:, 0].max(), bands)
    shells = np.linspace(0.0, edge.max(), 31 if mesh.topology.dim == 3 else 13)

    rows = []
    for a, b in itertools.pairwise(band_edges):
        inside = (centroid[:, 0] >= a) & (centroid[:, 0] < b)
        profile = rebin(radius[inside], heating[inside], case.geom.cells.volume[inside], shells)
        rows.append(Band(float(a), float(b), profile, float(tangent_slab(kappa, lo, np.array([a, b]))[0])))
    return shells, rows


def cell_error(case: Case, heating: np.ndarray, kappa: float) -> float:
    """Give the volume-weighted RMS error over every cell, against the point solution.

    Parameters
    ----------
    case
        The case that was traced.
    heating
        The volumetric heating of each cell.
    kappa
        The absorption coefficient of the gas.

    Returns
    -------
    error
        The error as a fraction of `case.mean_heating`.
    """
    centroid, vertices = case.mesh.geometry.cells.centroid, case.mesh.geometry.vertices
    exact = tangent_slab_point(kappa, vertices[:, 0].min(), centroid[:, 0])
    volume = case.geom.cells.volume
    return float(np.sqrt(np.sum(volume * (heating - exact) ** 2) / volume.sum())) / case.mean_heating


def profile_error(case: Case, heating: np.ndarray, kappa: float, bins: int = 26) -> float:
    """Give the RMS error over the axial bins, against their mean solution.

    Parameters
    ----------
    case
        The case that was traced.
    heating
        The volumetric heating of each cell.
    kappa
        The absorption coefficient of the gas.
    bins
        The number of bin edges.

    Returns
    -------
    error
        The error as a fraction of `case.mean_heating`.
    """
    _, profile, exact = axial_profile(case, heating, kappa, bins)
    return float(np.sqrt(np.mean((profile - exact) ** 2))) / case.mean_heating
