"""Write example results as the data files the Typst figures read.

Line data leaves as CSV. A field leaves as a bare PNG of its fills alone, with a
JSON sidecar that gives the `(xmin, ymin, xmax, ymax)` viewbox the image covers
and the colorbar or legend to draw around it.
"""

import json
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from raytrax.elements import ElementType
from raytrax.io import vtk

type Bounds = tuple[float, float, float, float]

# Enough samples that a Typst gradient through them is indistinguishable from the colormap.
COLORMAP_STOPS = 32


def save_csv(path: Path, columns: Sequence[np.ndarray], header: str) -> Path:
    """Write equal-length columns as one CSV with a header row.

    Parameters
    ----------
    path
        The file to write.
    columns
        The columns, in the order of `header`.
    header
        The comma-separated column names.

    Returns
    -------
    path
        The file written.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(path, np.column_stack(columns), delimiter=",", header=header, comments="", fmt="%.8g")
    return path


def save_json(path: Path, data: dict[str, Any]) -> Path:
    """Write a dictionary as indented JSON.

    Parameters
    ----------
    path
        The file to write.
    data
        The contents.

    Returns
    -------
    path
        The file written.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n")
    return path


def viewbox(bounds: Bounds) -> dict[str, float]:
    """Give the bounds as the sidecar keys the Typst `field` reads.

    Parameters
    ----------
    bounds
        The `(xmin, ymin, xmax, ymax)` the image covers.

    Returns
    -------
    viewbox
        The bounds keyed by name.
    """
    xmin, ymin, xmax, ymax = bounds
    return {"xmin": float(xmin), "ymin": float(ymin), "xmax": float(xmax), "ymax": float(ymax)}


def colorbar(low: float, high: float, cmap: str, label: str) -> dict[str, Any]:
    """Give the sidecar entry the Typst `colorbar` reads.

    Parameters
    ----------
    low
        The value at the bottom of the colormap.
    high
        The value at the top of the colormap.
    cmap
        The name of the matplotlib colormap the image was drawn with.
    label
        The axis label of the colorbar.

    Returns
    -------
    colorbar
        The range, label and colormap, with the colormap sampled as hex stops.
    """
    colours = mpl.colormaps[cmap](np.linspace(0.0, 1.0, COLORMAP_STOPS))
    return {
        "min": float(low),
        "max": float(high),
        "cmap": cmap,
        "stops": [mpl.colors.to_hex(colour) for colour in colours],
        "label": label,
    }


def bare_axes(bounds: Bounds, pixels: int = 1600) -> tuple[Figure, Axes]:
    """Make an axes with no decoration whose canvas maps exactly onto `bounds`.

    Parameters
    ----------
    bounds
        The `(xmin, ymin, xmax, ymax)` the canvas covers.
    pixels
        The length of the longer side of the image.

    Returns
    -------
    figure
        The figure to save.
    axes
        The axes to draw in, in data coordinates.
    """
    xmin, ymin, xmax, ymax = bounds
    dpi = 200
    scale = pixels / max(xmax - xmin, ymax - ymin)
    fig = plt.figure(figsize=((xmax - xmin) * scale / dpi, (ymax - ymin) * scale / dpi), dpi=dpi)
    ax = fig.add_axes((0, 0, 1, 1))
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_aspect("equal")
    ax.axis("off")
    return fig, ax


def save_image(stem: Path, fig: Figure, bounds: Bounds, **metadata: Any) -> list[Path]:
    """Write a bare figure as a transparent PNG and its JSON sidecar, and close it.

    Parameters
    ----------
    stem
        The path of both files, without a suffix.
    fig
        The figure made by `bare_axes`.
    bounds
        The `(xmin, ymin, xmax, ymax)` the figure covers.
    metadata
        More sidecar entries, such as `colorbar` or `legend`.

    Returns
    -------
    paths
        The image and the sidecar.
    """
    stem.parent.mkdir(parents=True, exist_ok=True)
    image = stem.with_suffix(".png")
    fig.savefig(image, transparent=True)
    plt.close(fig)
    return [image, save_json(stem.with_suffix(".json"), {**viewbox(bounds), **metadata})]


def cell_polygons(mesh: vtk.Mesh) -> Iterator[tuple[ElementType, np.ndarray, np.ndarray]]:
    """Give the cells of a 2D mesh as polygons, one element block at a time.

    Parameters
    ----------
    mesh
        The mesh to draw.

    Yields
    ------
    element
        The element type of the block.
    rows
        The cell rows of the block.
    corners
        The corner coordinates of each cell in the block.
    """
    vertices = mesh.geometry.vertices
    for index, element in enumerate(mesh.topology.block_elements):
        rows = np.flatnonzero(mesh.topology.cell_block == index)
        yield element, rows, vertices[mesh.topology.cell_vertices[rows][:, : element.num_vertices]]
