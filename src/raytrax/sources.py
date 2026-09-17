from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
from jax.typing import ArrayLike
from jaxtyping import Array, Float, Int

from raytrax.mesh.topology import EXTERIOR
from raytrax.paths import Path, RayState
from raytrax.scene import Scene, Simplices
from raytrax.shapes import Index, KeyArray, Vector


def sample_point(
    key: KeyArray, vertices: Float[Array, "nvert ndim"], simplices: Simplices, row: Index, path: Path[Any]
) -> Vector:
    """Sample a point uniformly in one cell or face.

    Parameters
    ----------
    key
        The PRNG key for the sample.
    vertices
        The position of each mesh vertex.
    simplices
        The simplex decomposition of each cell or face.
    row
        The index of the cell or face in `simplices`.
    path
        The ray kinematics, which sets the point density.

    Returns
    -------
    point
        A position in the cell or face.
    """
    key_simplex, key_barycentric = jax.random.split(key)
    index = jax.random.choice(key_simplex, simplices.weight.shape[1], p=simplices.weight[row])
    corners = vertices[simplices.vertices[row, index]]
    if simplices.apex is not None:
        corners = jnp.concatenate([simplices.apex[row][None], corners])
    return path.simplex_point(key_barycentric, corners)


class Sources(eqx.Module):
    """The emitted power of the cells and of the boundary faces.

    Attributes
    ----------
    cell_power
        The power that each cell emits.
    face_power
        The power that each emitting face emits, in the same units as `cell_power`.
    face
        The face id of each emitting face.
    cumulative
        The cumulative distribution over the cells, then the emitting faces.
    total
        The sum of all emitted power.
    """

    cell_power: Float[Array, " ncell"]
    face_power: Float[Array, " nemit"]
    face: Int[Array, " nemit"]
    cumulative: Float[Array, " nemitter"]
    total: Float[Array, ""]


def build(cell_power: ArrayLike, face_power: ArrayLike | None = None, face: ArrayLike | None = None) -> Sources:
    """Make the emission distribution from the power of cells and faces.

    Parameters
    ----------
    cell_power
        The power that each cell emits.
    face_power
        The power that each emitting face emits, in the same units as
        `cell_power`. None for no surface emission.
    face
        The face id of each entry of `face_power`. Each face must be on the boundary.

    Returns
    -------
    sources
        The emission distribution.
    """
    cell_power = jnp.asarray(cell_power)
    face_power = jnp.zeros(0) if face_power is None else jnp.asarray(face_power)
    face = jnp.zeros(0, dtype=int) if face is None else jnp.asarray(face)

    power = jnp.concatenate([cell_power, face_power])
    total = power.sum()
    return Sources(
        cell_power=cell_power,
        face_power=face_power,
        face=face,
        cumulative=jnp.cumulative_sum(power) / total,
        total=total,
    )


def sample(key: KeyArray, sources: Sources, scene: Scene, path: Path[Any]) -> tuple[RayState, Index]:
    """Sample one ray from the volume and surface emission.

    A ray from a cell has an isotropic direction. A ray from a face has a
    cosine-weighted direction about the inward normal.

    Parameters
    ----------
    key
        The PRNG key for the sample.
    sources
        The emission distribution.
    scene
        The device arrays of the mesh.
    path
        The ray kinematics of the run.

    Returns
    -------
    state
        The new ray, with zero travel.
    cell
        The cell that contains the start position of the ray.
    """
    key_source, key_point, key_direction = jax.random.split(key, 3)
    num_cells = sources.cell_power.shape[0]

    draw = jax.random.uniform(key_source)
    # `searchsorted` is int32 even under x64, and the march compares this
    # against the scene's own index arrays, so pin it to their width.
    source = jnp.minimum(jnp.searchsorted(sources.cumulative, draw), sources.cumulative.shape[0] - 1)
    source = source.astype(scene.cells.neighbour.dtype)

    # Traced under a static shape, so a volume-only problem never indexes into
    # the empty surface arrays.
    if sources.face_power.shape[0] == 0:
        return path.launch(
            sample_point(key_point, scene.vertices, scene.cells.interior, source, path),
            path.isotropic(key_direction),
            jnp.zeros(()),
        ), source

    from_surface = source >= num_cells
    cell = jnp.where(from_surface, 0, source)
    face = sources.face[jnp.where(from_surface, source - num_cells, 0)]

    # The stored normal points out of face_cells[:, 0], so the EXTERIOR column
    # says which way is back into the domain.
    cells = scene.faces.cells[face]
    inward = jnp.where(cells[0] == EXTERIOR, 1.0, -1.0) * scene.faces.normal[face]

    return path.launch(
        jnp.where(
            from_surface,
            sample_point(key_point, scene.vertices, scene.faces.surface, face, path),
            sample_point(key_point, scene.vertices, scene.cells.interior, cell, path),
        ),
        jnp.where(
            from_surface,
            path.lambertian(key_direction, inward),
            path.isotropic(key_direction),
        ),
        jnp.zeros(()),
    ), jnp.where(from_surface, cells.max(), cell)
