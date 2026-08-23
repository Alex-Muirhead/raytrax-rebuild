from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
from jax.typing import ArrayLike
from jaxtyping import Array, Float, Int

from raytrax.mesh.topology import EXTERIOR
from raytrax.paths import Path, RayState
from raytrax.scene import Scene
from raytrax.shapes import Index, KeyArray, Vector


def sample_cell_point(key: KeyArray, scene: Scene, cell: Index, path: Path[Any]) -> Vector:
    """Sample a point inside a cell, via its simplex decomposition."""
    key_simplex, key_barycentric = jax.random.split(key)
    index = jax.random.choice(key_simplex, scene.simplex_weight.shape[1], p=scene.simplex_weight[cell])
    corners = jnp.concatenate([scene.seed[cell][None], scene.vertices[scene.simplex_vertices[cell, index]]])
    return path.simplex_point(key_barycentric, corners)


def sample_face_point(key: KeyArray, scene: Scene, face: Index, path: Path[Any]) -> Vector:
    """Sample a point on a face, via its fan decomposition."""
    key_fan, key_barycentric = jax.random.split(key)
    index = jax.random.choice(key_fan, scene.face_fan_weight.shape[1], p=scene.face_fan_weight[face])
    return path.simplex_point(key_barycentric, scene.vertices[scene.face_fan_vertices[face, index]])


class Sources(eqx.Module):
    """Unnormalised emitted power from cell volumes and from boundary faces.

    Both are in the same units, so a single draw over `cumulative` picks between
    them.
    """

    cell_power: Float[Array, " ncell"]
    face_power: Float[Array, " nemit"]
    face: Int[Array, " nemit"]
    cumulative: Float[Array, " nemitter"]
    total: Float[Array, ""]


def build(cell_power: ArrayLike, face_power: ArrayLike | None = None, face: ArrayLike | None = None) -> Sources:
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
    """Draw one ray from the combined volume and surface emission.

    Volume emission is isotropic; surface emission is cosine-weighted about the
    inward normal.
    """
    key_source, key_point, key_direction = jax.random.split(key, 3)
    num_cells = sources.cell_power.shape[0]

    draw = jax.random.uniform(key_source)
    # `searchsorted` is int32 even under x64, and the march compares this
    # against the scene's own index arrays, so pin it to their width.
    source = jnp.minimum(jnp.searchsorted(sources.cumulative, draw), sources.cumulative.shape[0] - 1)
    source = source.astype(scene.neighbour.dtype)

    # Traced under a static shape, so a volume-only problem never indexes into
    # the empty surface arrays.
    if sources.face_power.shape[0] == 0:
        return path.launch(
            sample_cell_point(key_point, scene, source, path),
            path.isotropic(key_direction),
            jnp.zeros(()),
        ), source

    from_surface = source >= num_cells
    cell = jnp.where(from_surface, 0, source)
    face = sources.face[jnp.where(from_surface, source - num_cells, 0)]

    # The stored normal points out of face_cells[:, 0], so the EXTERIOR column
    # says which way is back into the domain.
    cells = scene.face_cells[face]
    inward = jnp.where(cells[0] == EXTERIOR, 1.0, -1.0) * scene.face_normal[face]

    return path.launch(
        jnp.where(
            from_surface,
            sample_face_point(key_point, scene, face, path),
            sample_cell_point(key_point, scene, cell, path),
        ),
        jnp.where(
            from_surface,
            path.lambertian(key_direction, inward),
            path.isotropic(key_direction),
        ),
        jnp.zeros(()),
    ), jnp.where(from_surface, cells.max(), cell)
