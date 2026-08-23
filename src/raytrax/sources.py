import equinox as eqx
import jax
import jax.numpy as jnp
from jax.typing import ArrayLike
from jaxtyping import Array, Float, Int

from raytrax.mesh.topology import EXTERIOR
from raytrax.paths import Path, RayState
from raytrax.random import simplex
from raytrax.scene import Scene
from raytrax.shapes import Index, KeyArray, Vector


def sample_cell_point(key: KeyArray, scene: Scene, cell: Index) -> Vector:
    """Sample a point uniformly inside a cell, via its simplex decomposition."""
    key_simplex, key_barycentric = jax.random.split(key)
    index = jax.random.choice(key_simplex, scene.simplex_weight.shape[1], p=scene.simplex_weight[cell])
    corners = scene.vertices[scene.simplex_vertices[cell, index]]
    weights = simplex(key_barycentric, scene.dim)
    return weights[0] * scene.seed[cell] + weights[1:] @ corners


def sample_face_point(key: KeyArray, scene: Scene, face: Index) -> Vector:
    """Sample a point uniformly on a face, via its fan decomposition."""
    key_fan, key_barycentric = jax.random.split(key)
    index = jax.random.choice(key_fan, scene.face_fan_weight.shape[1], p=scene.face_fan_weight[face])
    corners = scene.vertices[scene.face_fan_vertices[face, index]]
    return simplex(key_barycentric, scene.dim - 1) @ corners


class Sources(eqx.Module):
    """Unnormalised emitted power from cell volumes and from boundary faces.

    Both are in the same units so a single draw can choose between them. The
    cumulative distribution is precomputed: drawing by `searchsorted` is
    logarithmic in the number of emitters, where weighted `choice` is linear,
    and refilling dead lanes redraws often enough for that to dominate.
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


def sample(key: KeyArray, sources: Sources, scene: Scene, path: Path) -> tuple[RayState, Index]:
    """Draw one ray from the combined volume and surface emission.

    Volume emission is isotropic; surface emission is cosine-weighted about the
    inward normal. Both go through `path`, which draws in 3D and keeps the part
    that lies in the mesh -- so on a 2D mesh the tangent comes back shorter than
    unit rather than confined to the plane.
    """
    key_source, key_point, key_direction = jax.random.split(key, 3)
    num_cells = sources.cell_power.shape[0]

    draw = jax.random.uniform(key_source)
    source = jnp.minimum(jnp.searchsorted(sources.cumulative, draw), sources.cumulative.shape[0] - 1)

    # Shapes are static, so a volume-only problem never pays for -- or indexes
    # into -- the surface branch.
    if sources.face_power.shape[0] == 0:
        return RayState(
            position=sample_cell_point(key_point, scene, source),
            tangent=path.isotropic(key_direction, scene.dim),
            travel=jnp.zeros(()),
        ), source

    from_surface = source >= num_cells
    cell = jnp.where(from_surface, 0, source)
    face = sources.face[jnp.where(from_surface, source - num_cells, 0)]

    # The stored normal points out of face_cells[:, 0]; whichever column is
    # EXTERIOR tells us which way is back into the domain.
    cells = scene.face_cells[face]
    inward = jnp.where(cells[0] == EXTERIOR, 1.0, -1.0) * scene.face_normal[face]

    return RayState(
        position=jnp.where(
            from_surface,
            sample_face_point(key_point, scene, face),
            sample_cell_point(key_point, scene, cell),
        ),
        tangent=jnp.where(
            from_surface,
            path.lambertian(key_direction, inward),
            path.isotropic(key_direction, scene.dim),
        ),
        travel=jnp.zeros(()),
    ), jnp.where(from_surface, cells.max(), cell)
