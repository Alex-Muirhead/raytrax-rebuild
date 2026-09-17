"""Small hand-built meshes. `mixed_3d` is the one that matters: it is the only
fixture where cells of different types share faces, which is what the padded
half-space layout exists to support."""

import jax.numpy as jnp
import numpy as np

from raytrax import interactions
from raytrax import scene as scene_module
from raytrax.interactions import Kind
from raytrax.mesh import boundaries, geometry, topology

CUBE = np.array(
    [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0], [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]],
    dtype=float,
)
SQUARE = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=float)


def make(blocks, vertices, groups=None):
    mesh_topology = topology.build(blocks)
    mesh_geometry = geometry.build(mesh_topology, vertices)
    tagged = boundaries.tag(mesh_topology, groups or {})
    return mesh_topology, mesh_geometry, scene_module.build(mesh_topology, mesh_geometry, tagged)


def unit_hex():
    return [("hexahedron", np.arange(8)[None, :])], CUBE


def unit_tet():
    return [("tetra", np.array([[0, 1, 3, 4]]))], CUBE


def unit_wedge():
    return [("wedge", np.array([[0, 1, 3, 4, 5, 7]]))], CUBE


def unit_pyramid():
    return [("pyramid", np.array([[0, 1, 2, 3, 8]]))], np.vstack([CUBE, [[0.5, 0.5, 1.0]]])


def unit_quad():
    return [("quad", np.arange(4)[None, :])], SQUARE


def unit_triangle():
    return [("triangle", np.array([[0, 1, 2]]))], SQUARE


ALL_ELEMENTS = {
    "hexahedron": unit_hex,
    "tetra": unit_tet,
    "wedge": unit_wedge,
    "pyramid": unit_pyramid,
    "quad": unit_quad,
    "triangle": unit_triangle,
}


def mixed_3d():
    """A hex, a pyramid on its top face, and a tet on one of the pyramid's."""
    vertices = np.vstack([CUBE, [[0.5, 0.5, 2.0], [2.0, 0.5, 2.0]]])
    blocks = [
        ("hexahedron", np.array([[0, 1, 2, 3, 4, 5, 6, 7]])),
        ("pyramid", np.array([[4, 5, 6, 7, 8]])),
        ("tetra", np.array([[5, 6, 8, 9]])),
    ]
    return blocks, vertices


def brick(shape, dim=3):
    """A regular grid of unit hexes (3D) or quads (2D), tagged on each side."""
    counts = np.array(shape)
    grid = np.indices(tuple(counts + 1)).reshape(dim, -1).T.astype(float)
    ids = np.arange(len(grid)).reshape(counts + 1)

    if dim == 3:
        corner = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0], [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]])
        name = "hexahedron"
    else:
        corner = np.array([[0, 0], [1, 0], [1, 1], [0, 1]])
        name = "quad"

    origins = np.indices(tuple(counts)).reshape(dim, -1).T
    cells = ids[tuple((origins[:, None, :] + corner[None, :, :]).transpose(2, 0, 1))]
    return [(name, cells)], grid


def brick_sides(mesh_topology, mesh_geometry, vertices):
    """Face groups for the six (or four) outer sides of a brick."""
    boundary = np.flatnonzero(mesh_topology.is_boundary)
    centre = mesh_geometry.faces.centroid[boundary]
    names = ["xlo", "xhi", "ylo", "yhi", "zlo", "zhi"][: 2 * mesh_topology.dim]

    groups = {}
    for axis in range(mesh_topology.dim):
        for side, limit in enumerate((vertices[:, axis].min(), vertices[:, axis].max())):
            groups[names[2 * axis + side]] = boundary[np.isclose(centre[:, axis], limit)]
    return groups


def uniform_boundary(kind, reflectivity=1.0):
    """One implicit group covering every boundary face."""
    return interactions.Interactions(kind=(Kind(kind),), reflectivity=jnp.array([reflectivity]))
