"""Off-screen pyvista renders of a mesh and a cell field.

Needs the `visualisation` dependency group, and a display: a headless machine
needs `xvfb-run`, or the `vtk-osmesa` wheel in place of `vtk`.
"""

import os
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

from raytrax.io import vtk
from raytrax_examples import colours
from raytrax_examples.export import Bounds

# Must be set before pyvista imports vtk, and before any Plotter is built.
os.environ.setdefault("PYVISTA_OFF_SCREEN", "true")
import pyvista as pv

# Never negotiate a jupyter backend: marimo does not render ipywidgets, so
# every figure is a screenshot taken off screen.
pv.OFF_SCREEN = True
pv.global_theme.notebook = False

VTK_CELL = {
    "triangle": pv.CellType.TRIANGLE,
    "quad": pv.CellType.QUAD,
    "tetra": pv.CellType.TETRA,
    "hexahedron": pv.CellType.HEXAHEDRON,
    "wedge": pv.CellType.WEDGE,
    "pyramid": pv.CellType.PYRAMID,
}

# The cut face points along +y, so the camera sits on that side to see it,
# tilted towards the hot end so the gradient runs across. Position, focal
# point, view up -- pyvista's camera_position format.
CYLINDER_CAMERA = ((-0.9, 2.5, 1.0), (0.0, 0.0, 0.0), (0.0, 0.0, 1.0))


def to_pyvista(mesh: vtk.Mesh, values: np.ndarray | None = None) -> pv.UnstructuredGrid:
    """Wrap a mesh as a pyvista grid, cell for cell in the same order.

    Parameters
    ----------
    mesh
        The mesh to wrap.
    values
        A cell field to attach as `heating`.

    Returns
    -------
    grid
        The pyvista grid.
    """
    topology = mesh.topology
    cells, types = [], []
    for index, element in enumerate(topology.block_elements):
        rows = np.flatnonzero(topology.cell_block == index)
        connectivity = topology.cell_vertices[rows, : element.num_vertices]
        counts = np.full((len(rows), 1), element.num_vertices)
        cells.append(np.hstack([counts, connectivity]).ravel())
        types.append(np.full(len(rows), VTK_CELL[element.name], dtype=np.uint8))

    # A 2D mesh stores two columns; VTK always wants three.
    points = np.zeros((len(mesh.geometry.vertices), 3))
    points[:, : topology.dim] = mesh.geometry.vertices

    grid = pv.UnstructuredGrid(np.concatenate(cells), np.concatenate(types), points)
    if values is not None:
        grid.cell_data["heating"] = values
    return grid


def half_cut(grid: pv.UnstructuredGrid, ragged: bool = True) -> pv.DataSet:
    """Keep the y <= 0 half of a grid.

    Parameters
    ----------
    grid
        The grid to cut.
    ragged
        Whether to keep whole cells, rather than slice through them.

    Returns
    -------
    half
        The kept half.
    """
    if not ragged:
        return grid.clip(normal="y", origin=(0.0, 0.0, 0.0))
    # Any vertex at y <= 0 keeps the whole cell, so the cut face is the cell
    # boundary itself: stair-stepped, and it bulges past y = 0.
    return grid.extract_points(grid.points[:, 1] <= 0.0, adjacent_cells=True)


def mesh_extent(mesh: vtk.Mesh, margin: float = 0.04) -> Bounds:
    """Give the padded bounds of the mesh vertices in x and y.

    Parameters
    ----------
    mesh
        The mesh to measure.
    margin
        The padding, as a fraction of each side.

    Returns
    -------
    bounds
        The `(xmin, ymin, xmax, ymax)` of the vertices.
    """
    vertices = mesh.geometry.vertices[:, :2]
    (xmin, ymin), (xmax, ymax) = vertices.min(axis=0), vertices.max(axis=0)
    pad_x, pad_y = (xmax - xmin) * margin, (ymax - ymin) * margin
    return float(xmin - pad_x), float(ymin - pad_y), float(xmax + pad_x), float(ymax + pad_y)


def fit_window(bounds: Bounds, pixels: int = 1600) -> tuple[int, int]:
    """Give a window size in pixels with the aspect ratio of `bounds`.

    Parameters
    ----------
    bounds
        The `(xmin, ymin, xmax, ymax)` the window shows.
    pixels
        The length of the longer side.

    Returns
    -------
    size
        The width and height in pixels.
    """
    xmin, ymin, xmax, ymax = bounds
    width, height = xmax - xmin, ymax - ymin
    scale = pixels / max(width, height)
    return max(round(width * scale), 1), max(round(height * scale), 1)


def projected_extent(mesh: vtk.Mesh, camera_position: Any, margin: float = 0.04) -> Bounds:
    """Give the padded bounds of the mesh vertices on the view plane of a camera.

    A tilted camera's visible frame is not axis-aligned in `(x, y, z)`, so a
    window fitted to the 3D bounding box leaves a loose margin.

    Parameters
    ----------
    mesh
        The mesh to measure.
    camera_position
        The camera position, focal point and view up.
    margin
        The padding, as a fraction of each side.

    Returns
    -------
    bounds
        The `(umin, vmin, umax, vmax)` of the vertices along the camera's right and up.
    """
    position, focal, up = (np.array(v, dtype=float) for v in camera_position)
    forward = focal - position
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, up)
    right /= np.linalg.norm(right)
    up = np.cross(right, forward)

    u = mesh.geometry.vertices @ right
    v = mesh.geometry.vertices @ up
    pad_u, pad_v = (u.max() - u.min()) * margin, (v.max() - v.min()) * margin
    return float(u.min() - pad_u), float(v.min() - pad_v), float(u.max() + pad_u), float(v.max() + pad_v)


def render(
    build: Callable[[pv.Plotter], Any],
    path: Path | None = None,
    size: tuple[int, int] = (1400, 900),
    scale: int = 2,
    transparent: bool = True,
) -> np.ndarray:
    """Draw a scene off screen, and return the frame.

    Parameters
    ----------
    build
        The function that adds the scene to a plotter.
    path
        The PNG file to write the frame to, if any.
    size
        The window size in pixels.
    scale
        The resolution multiplier of the screenshot.
    transparent
        Whether the background is transparent.

    Returns
    -------
    image
        The frame as an RGBA array.
    """
    plotter = pv.Plotter(off_screen=True, window_size=list(size))
    try:
        build(plotter)
        return plotter.screenshot(
            None if path is None else str(path),
            return_img=True,
            scale=scale,
            transparent_background=transparent,
        )
    finally:
        # marimo re-runs a cell on any upstream change, and a Plotter left
        # open holds its render window.
        plotter.close()


def bar_args(label: str) -> dict[str, Any]:
    """Give the scalar bar arguments that place it in the right-hand margin.

    Parameters
    ----------
    label
        The title of the bar.

    Returns
    -------
    arguments
        The pyvista `scalar_bar_args`.
    """
    return {
        "title": label,
        "vertical": True,
        "color": "black",
        "title_font_size": 20,
        "label_font_size": 16,
        "position_x": 0.87,
        "position_y": 0.25,
        "width": 0.04,
        "height": 0.5,
    }


def frame(plotter: pv.Plotter, three_d: bool, tight_bounds: Bounds | None = None) -> None:
    """Fit the camera to the scene.

    Parameters
    ----------
    plotter
        The plotter that holds the scene.
    three_d
        Whether to use the tilted `CYLINDER_CAMERA`, rather than face on.
    tight_bounds
        The bounds to fit exactly, with no margin for a scalar bar. Face on,
        world `(xmin, ymin, xmax, ymax)`; tilted, the `projected_extent`.
    """
    if tight_bounds is not None and not three_d:
        xmin, ymin, xmax, ymax = tight_bounds
        cx, cy = (xmin + xmax) / 2, (ymin + ymax) / 2
        plotter.camera_position = [(cx, cy, 1.0), (cx, cy, 0.0), (0.0, 1.0, 0.0)]
        plotter.camera.parallel_scale = (ymax - ymin) / 2
        return
    if three_d:
        plotter.camera_position = list(CYLINDER_CAMERA)
    else:
        plotter.view_xy()
    # Called for its centring; its scale fits the bounding-box diagonal, which
    # is looser than the footprint, so a tight fit overwrites it.
    plotter.reset_camera()
    if tight_bounds is not None:
        _, vmin, _, vmax = tight_bounds
        plotter.camera.parallel_scale = (vmax - vmin) / 2
    else:
        plotter.camera.zoom(0.92)


def draw(
    plotter: pv.Plotter,
    mesh: vtk.Mesh,
    values: np.ndarray | None = None,
    ragged: bool = True,
    cmap: str | list[str] = colours.SEQUENTIAL,
    label: str = "Volumetric heating",
    bar: bool = True,
    tight_bounds: Bounds | None = None,
    symmetric: bool = False,
) -> None:
    """Add a mesh to a plotter, cut in half if it is 3D and face on if it is 2D.

    Parameters
    ----------
    plotter
        The plotter to add to.
    mesh
        The mesh to draw.
    values
        The cell field to colour by, or None for a flat grey.
    ragged
        Whether a 3D cut keeps whole cells.
    cmap
        The colormap of the field.
    label
        The title of the scalar bar.
    bar
        Whether to draw the scalar bar.
    tight_bounds
        The bounds to fit exactly, as `frame` takes them.
    symmetric
        Whether the colour range is centred on zero, for a signed field.
    """
    grid = to_pyvista(mesh, values)
    if values is None:
        colours: dict[str, Any] = {"color": "#d9d9d9"}
    else:
        # Pinned to the full field, not the piece pyvista actually draws:
        # `half_cut` in 3D would otherwise narrow the range to whatever
        # survives the cut, and the exported colour bar range would lie.
        if symmetric:
            bound = float(np.nanmax(np.abs(values)))
            clim = (-bound, bound)
        else:
            clim = (float(np.nanmin(values)), float(np.nanmax(values)))
        colours = {"scalars": "heating", "cmap": cmap, "show_scalar_bar": bar, "clim": clim}
        if bar:
            colours["scalar_bar_args"] = bar_args(label)

    # Before any camera work: the parallel scale is set when projection is
    # switched, so switching afterwards throws the framing away.
    plotter.enable_parallel_projection()
    plotter.set_background("white")

    if mesh.topology.dim == 3:
        piece = half_cut(grid, ragged=ragged)
        assert values is None or "heating" in piece.cell_data, "cell field lost in extraction"
        plotter.add_mesh(piece, show_edges=True, edge_color="#262626", line_width=0.4, **colours)
        # A faint skin of the whole cylinder, so the cut reads as a cut.
        plotter.add_mesh(
            grid.extract_surface(algorithm="dataset_surface"),
            style="wireframe",
            color="#b0b0b0",
            line_width=0.4,
            opacity=0.2,
        )
    else:
        # Flat colour: with lighting on, VTK shades each polygon and the
        # colour stops reading off the colour bar.
        plotter.add_mesh(grid, show_edges=True, edge_color="#262626", line_width=1.0, lighting=False, **colours)

    frame(plotter, mesh.topology.dim == 3, tight_bounds=tight_bounds)


def draw_mesh(
    plotter: pv.Plotter,
    mesh: vtk.Mesh,
    ragged: bool = True,
    legend: bool = True,
    tight_bounds: Bounds | None = None,
) -> list[dict[str, str]] | None:
    """Add a bare mesh to a plotter, coloured by element type where there is more than one.

    Parameters
    ----------
    plotter
        The plotter to add to.
    mesh
        The mesh to draw.
    ragged
        Whether a 3D cut keeps whole cells.
    legend
        Whether to draw the legend in the image.
    tight_bounds
        The bounds to fit exactly, as `frame` takes them.

    Returns
    -------
    legend
        One label and colour per element type, or None if there is only one type.
    """
    blocks = mesh.topology.block_elements
    if len({element.name for element in blocks}) < 2:
        draw(plotter, mesh, ragged=ragged, tight_bounds=tight_bounds)
        return None

    # Ordered by vertex count, so a triangle always takes the first colour.
    names = [element.name for element in sorted(set(blocks), key=lambda element: element.num_vertices)]
    code = np.zeros(mesh.topology.num_cells)
    for index, element in enumerate(blocks):
        code[mesh.topology.cell_block == index] = names.index(element.name)

    # One flat colour per element type and a legend, not a ramp.
    palette = colours.CYCLE[: len(names)]
    draw(plotter, mesh, code, ragged, cmap=palette, label="", bar=False, tight_bounds=tight_bounds)
    entries = [{"label": name, "colour": colour} for name, colour in zip(names, palette, strict=True)]
    if legend:
        plotter.add_legend(
            [[entry["label"], entry["colour"]] for entry in entries],
            bcolor="white",
            border=True,
            size=(0.13, 0.09),
            loc="upper right",
        )
    return entries
