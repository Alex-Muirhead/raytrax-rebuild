import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Bool, Float, Int

from raytrax import interactions as interaction
from raytrax.interactions import Interactions
from raytrax.paths import Path, RayState
from raytrax.scene import Scene
from raytrax.sources import Sources
from raytrax.sources import sample as sample_source


def _keep(mask, new, old):
    """Advance only the lanes that are still alive, broadcasting the mask."""
    return jnp.where(mask.reshape(mask.shape + (1,) * (old.ndim - mask.ndim)), new, old)


class Deposit(eqx.Module):
    """Where the energy ended up."""

    cells: Float[Array, " ncell"]
    faces: Float[Array, " nface"]


class Launch(eqx.Module):
    """Refill dead lanes from a source until a launch budget is spent.

    A step costs O(lanes), not O(alive), so a lane whose ray has terminated is
    pure waste. Refilling keeps the batch full instead of draining it, which
    turns the occupancy fraction into the speedup.

    Every launched ray carries exactly `quantum`, so the deposited total stays
    `sources.total` however the launches happen to land across the lanes.
    """

    sources: Sources
    budget: Int[Array, ""]
    quantum: Float[Array, ""]
    interval: int = eqx.field(static=True)


class Rays(eqx.Module):
    """A batch of rays part-way through the mesh."""

    cell: Int[Array, " nray"]
    state: RayState
    energy: Float[Array, " nray"]
    alive: Bool[Array, " nray"]


def _refill(key, rays, launched, launch: Launch, scene: Scene):
    """Overwrite dead lanes with fresh rays, up to the remaining budget."""
    dead = ~rays.alive
    # Rank among dead lanes, so the budget is spent on a well-defined prefix.
    rank = jnp.cumulative_sum(dead) - 1
    take = dead & (launched + rank < launch.budget)

    keys = jax.random.split(key, rays.cell.shape[0])
    state, cell = jax.vmap(sample_source, in_axes=(0, None, None))(keys, launch.sources, scene)

    return Rays(
        cell=_keep(take, cell, rays.cell),
        state=jax.tree.map(lambda new, old: _keep(take, new, old), state, rays.state),
        energy=_keep(take, jnp.full_like(rays.energy, launch.quantum), rays.energy),
        alive=rays.alive | take,
    ), launched + jnp.sum(take)


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
    launch: Launch | None = None,
) -> tuple[Rays, Deposit]:
    """Walk rays until they terminate or `max_steps` runs out.

    Exiting on `any(alive)` makes `max_steps` a safety net rather than a cost:
    the batch pays for its longest ray, not for the bound. It does not make dead
    rays free -- a step costs O(lanes), not O(alive) -- so pass a `Launch` to
    refill dead lanes and keep the batch full.
    """

    def body(carry):
        index, key, launched, rays, deposit = carry
        key, subkey, refill_key = jax.random.split(key, 3)
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

        if launch is not None:
            # Outside the vmap, so this is a real branch and costs nothing on
            # the steps it skips.
            rays, launched = jax.lax.cond(
                (index % launch.interval == 0) & (launched < launch.budget),
                lambda: _refill(refill_key, rays, launched, launch, scene),
                lambda: (rays, launched),
            )
        return index + 1, key, launched, rays, deposit

    def cond(carry):
        index, _, launched, rays, _ = carry
        pending = launched < launch.budget if launch is not None else False
        return (index < max_steps) & (jnp.any(rays.alive) | pending)

    deposit = Deposit(cells=jnp.zeros(scene.num_cells), faces=jnp.zeros(scene.num_faces))
    launched = jnp.sum(rays.alive)
    _, _, _, rays, deposit = jax.lax.while_loop(cond, body, (0, key, launched, rays, deposit))
    return rays, deposit


def trace(
    key,
    scene: Scene,
    path: Path,
    interactions: Interactions,
    absorptivity: Float[Array, " ncell"],
    sources: Sources,
    num_rays: int,
    lanes: int | None = None,
    max_steps: int = 100_000,
    refill_interval: int = 8,
) -> tuple[Rays, Deposit]:
    """Trace `num_rays` rays through `lanes` lanes, refilling as they terminate.

    `lanes` trades memory against occupancy; it defaults to tracing everything
    at once, which leaves lanes idle as rays finish. Refilling every step costs
    more in sampling than it recovers in occupancy, so `refill_interval` lets
    lanes sit briefly idle instead -- 8 measured best on a shock-layer case, but
    it is worth sweeping on a new problem.
    """
    lanes = min(num_rays, lanes or num_rays)
    key_launch, key_march = jax.random.split(key)

    state, cell = jax.vmap(sample_source, in_axes=(0, None, None))(jax.random.split(key_launch, lanes), sources, scene)
    quantum = sources.total / num_rays
    rays = Rays(cell=cell, state=state, energy=jnp.full(lanes, quantum), alive=jnp.ones(lanes, dtype=bool))

    launch = None
    if lanes < num_rays:
        launch = Launch(sources=sources, budget=jnp.asarray(num_rays), quantum=quantum, interval=refill_interval)
    return march(key_march, rays, scene, path, interactions, absorptivity, max_steps=max_steps, launch=launch)
