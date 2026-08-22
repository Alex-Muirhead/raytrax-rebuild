import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Bool, Float, Int

from raytrax import interactions as interaction
from raytrax.interactions import Interactions
from raytrax.paths import Path, RayState
from raytrax.scene import Scene


def _keep(mask, new, old):
    """Advance only the lanes that are still alive, broadcasting the mask."""
    return jnp.where(mask.reshape(mask.shape + (1,) * (old.ndim - mask.ndim)), new, old)


class Deposit(eqx.Module):
    """Where the energy ended up."""

    cells: Float[Array, " ncell"]
    faces: Float[Array, " nface"]


class Rays(eqx.Module):
    """A batch of rays part-way through the mesh."""

    cell: Int[Array, " nray"]
    state: RayState
    energy: Float[Array, " nray"]
    alive: Bool[Array, " nray"]


def step(key, scene: Scene, path: Path, interactions: Interactions, absorptivity, cell, state: RayState, energy):
    """Cross one cell, depositing continuously along the way."""
    normal, offset = scene.normal[cell], scene.offset[cell]
    slot, distance = path.crossing(normal, offset, state)
    # A corner or grazing hit can give a small negative distance; clamping keeps
    # the deposited energy non-negative without pushing the ray off the plane.
    distance = jnp.maximum(distance, 0.0)

    transmission = jnp.exp(-absorptivity[cell] * distance)
    cell_deposit = energy * (1 - transmission)
    energy = energy * transmission
    state = path.advance(state, distance)

    face = scene.face[cell, slot]
    neighbour = scene.neighbour[cell, slot]
    at_boundary = neighbour < 0

    group = jnp.where(at_boundary, scene.face_group[face], 0)
    hit_state, absorbed, terminated = interaction.apply(path, interactions, group, key, state, normal[slot])

    # An interior crossing just walks on; only a boundary consults the table.
    state = jax.tree.map(lambda new, old: jnp.where(at_boundary, new, old), hit_state, state)
    face_deposit = jnp.where(at_boundary, energy * absorbed, 0.0)

    return (
        jnp.where(at_boundary, cell, neighbour),
        state,
        energy - face_deposit,
        ~(at_boundary & terminated),
        cell_deposit,
        face,
        face_deposit,
    )


@eqx.filter_jit
def march(
    key,
    rays: Rays,
    scene: Scene,
    path: Path,
    interactions: Interactions,
    absorptivity: Float[Array, " ncell"],
    max_steps: int = 1000,
) -> tuple[Rays, Deposit]:
    """Walk a batch of rays until they all terminate or `max_steps` runs out.

    Exiting on `any(alive)` makes `max_steps` a safety net rather than a cost:
    the batch pays for its longest ray, not for the bound. It does not make dead
    rays free -- under `vmap` a step costs O(lanes), not O(alive) -- so the
    remaining waste is idle lanes, which only refilling them can recover.
    """

    def body(carry):
        index, key, rays, deposit = carry
        key, subkey = jax.random.split(key)
        was_alive = rays.alive

        cell, state, energy, alive, cell_deposit, face, face_deposit = jax.vmap(
            step, in_axes=(0, None, None, None, None, 0, 0, 0)
        )(
            jax.random.split(subkey, rays.cell.shape[0]),
            scene,
            path,
            interactions,
            absorptivity,
            rays.cell,
            rays.state,
            rays.energy,
        )

        # The scatter *must* be outside the vmap; inside it, XLA materialises a
        # full per-ray copy of the accumulator and memory use explodes.
        deposit = Deposit(
            cells=deposit.cells.at[rays.cell].add(jnp.where(was_alive, cell_deposit, 0.0)),
            faces=deposit.faces.at[face].add(jnp.where(was_alive, face_deposit, 0.0)),
        )
        rays = Rays(
            cell=_keep(was_alive, cell, rays.cell),
            state=jax.tree.map(lambda new, old: _keep(was_alive, new, old), state, rays.state),
            energy=_keep(was_alive, energy, rays.energy),
            alive=was_alive & alive,
        )
        return index + 1, key, rays, deposit

    def cond(carry):
        index, _, rays, _ = carry
        return (index < max_steps) & jnp.any(rays.alive)

    deposit = Deposit(cells=jnp.zeros(scene.num_cells), faces=jnp.zeros(scene.num_faces))
    _, _, rays, deposit = jax.lax.while_loop(cond, body, (0, key, rays, deposit))
    return rays, deposit
