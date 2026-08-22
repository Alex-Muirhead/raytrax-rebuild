import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, Int

from raytrax.mesh.topology import EXTERIOR
from raytrax.paths import RayState
from raytrax.random import cosine_hemisphere, simplex, sphere
from raytrax.scene import Scene


def sample_cell_point(key, scene: Scene, cell) -> Float[Array, " ndim"]:
    """Sample a point uniformly inside a cell, via its simplex decomposition."""
    key_simplex, key_barycentric = jax.random.split(key)
    index = jax.random.choice(key_simplex, scene.simplex_weight.shape[1], p=scene.simplex_weight[cell])
    corners = scene.vertices[scene.simplex_vertices[cell, index]]
    weights = simplex(key_barycentric, scene.dim)
    return weights[0] * scene.seed[cell] + weights[1:] @ corners


def sample_face_point(key, scene: Scene, face) -> Float[Array, " ndim"]:
    """Sample a point uniformly on a face, via its fan decomposition."""
    key_fan, key_barycentric = jax.random.split(key)
    index = jax.random.choice(key_fan, scene.face_fan_weight.shape[1], p=scene.face_fan_weight[face])
    corners = scene.vertices[scene.face_fan_vertices[face, index]]
    return simplex(key_barycentric, scene.dim - 1) @ corners


class Sources(eqx.Module):
    """Unnormalised emitted power from cell volumes and from boundary faces.

    Both are in the same units so a single draw can choose between them.
    """

    cell_power: Float[Array, " ncell"]
    face_power: Float[Array, " nemit"]
    face: Int[Array, " nemit"]

    @property
    def total(self) -> Float[Array, ""]:
        return self.cell_power.sum() + self.face_power.sum()


def sample(key, sources: Sources, scene: Scene) -> tuple[RayState, Int[Array, ""]]:
    """Draw one ray from the combined volume and surface emission.

    Volume emission is isotropic; surface emission is cosine-weighted about the
    inward normal.
    """
    key_source, key_point, key_direction = jax.random.split(key, 3)
    num_cells = sources.cell_power.shape[0]

    # Shapes are static, so a volume-only problem never pays for -- or indexes
    # into -- the surface branch.
    if sources.face_power.shape[0] == 0:
        cell = jax.random.choice(key_source, num_cells, p=sources.cell_power / sources.cell_power.sum())
        return RayState(
            position=sample_cell_point(key_point, scene, cell),
            tangent=sphere(key_direction, scene.dim),
            travel=jnp.zeros(()),
        ), cell

    power = jnp.concatenate([sources.cell_power, sources.face_power])
    source = jax.random.choice(key_source, power.shape[0], p=power / power.sum())
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
            cosine_hemisphere(key_direction, inward),
            sphere(key_direction, scene.dim),
        ),
        travel=jnp.zeros(()),
    ), jnp.where(from_surface, cells.max(), cell)
