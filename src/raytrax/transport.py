from typing import Any, NamedTuple

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Bool, Float, Int

from raytrax import interactions as interaction
from raytrax import tuning
from raytrax.interactions import Interactions
from raytrax.paths import Path, RayState
from raytrax.scene import Scene
from raytrax.shapes import Flag, Index, KeyArray, Scalar
from raytrax.sources import Sources
from raytrax.sources import sample as sample_source


def _keep(mask: Bool[Array, "..."], new: Array, old: Array) -> Array:
    """Select `new` where the mask is true, and `old` where it is false.

    Parameters
    ----------
    mask
        The selection over the leading axes of `old`.
    new
        The values where `mask` is true.
    old
        The values where `mask` is false.

    Returns
    -------
    selected
        The selected values, with the shape of `old`.
    """
    return jnp.where(mask.reshape(mask.shape + (1,) * (old.ndim - mask.ndim)), new, old)


def _keep_tree[T](mask: Bool[Array, "..."], new: T, old: T) -> T:
    """Select `new` where the mask is true, and `old` where it is false, for each leaf.

    Parameters
    ----------
    mask
        The selection over the leading axes of each leaf.
    new
        The pytree to use where `mask` is true.
    old
        The pytree to use where `mask` is false, with the same structure as `new`.

    Returns
    -------
    selected
        The pytree of selected leaves.
    """

    def keep(new_leaf: Array, old_leaf: Array) -> Array:
        return _keep(mask, new_leaf, old_leaf)

    return jax.tree.map(keep, new, old)


class Deposit(eqx.Module):
    """The energy that the rays deposit, and the energy that escapes.

    Attributes
    ----------
    cells
        The energy that each cell absorbs.
    faces
        The energy that each face absorbs.
    escaped
        The energy that leaves the domain.
    """

    cells: Float[Array, " ncell"]
    faces: Float[Array, " nface"]
    escaped: Scalar


class Launch(eqx.Module):
    """The parameters to refill dead lanes from a source during a march.

    Attributes
    ----------
    sources
        The emission distribution of the new rays.
    budget
        The total number of rays to launch, including the first batch.
    quantum
        The energy of each new ray.
    interval
        The number of steps between refills.
    """

    sources: Sources
    budget: Int[Array, ""]
    quantum: Float[Array, ""]
    interval: int = eqx.field(static=True)


class Rays(eqx.Module):
    """A batch of rays in the mesh, one for each lane.

    Attributes
    ----------
    cell
        The cell that contains each ray.
    state
        The position and direction of each ray.
    energy
        The energy that each ray carries.
    alive
        True for each lane that holds a ray in flight.
    """

    cell: Int[Array, " nray"]
    state: RayState
    energy: Float[Array, " nray"]
    alive: Bool[Array, " nray"]


@jax.named_scope("refill")
def _refill(
    key: KeyArray, rays: Rays, launched: Index, launch: Launch, scene: Scene, path: Path[Any]
) -> tuple[Rays, Index]:
    """Put new rays in dead lanes, up to the remaining launch budget.

    Parameters
    ----------
    key
        The PRNG key for the new rays.
    rays
        The batch of rays.
    launched
        The number of rays launched before this refill.
    launch
        The refill parameters.
    scene
        The device arrays of the mesh.
    path
        The ray kinematics of the run.

    Returns
    -------
    rays
        The batch with new rays in the refilled lanes.
    launched
        The number of rays launched after this refill.
    """
    dead = ~rays.alive
    # Rank among dead lanes, so the budget is spent on a well-defined prefix.
    rank = jnp.cumulative_sum(dead) - 1
    take = dead & (launched + rank < launch.budget)

    keys = jax.random.split(key, rays.cell.shape[0])
    state, cell = jax.vmap(sample_source, in_axes=(0, None, None, None))(keys, launch.sources, scene, path)

    return Rays(
        cell=_keep(take, cell, rays.cell),
        state=_keep_tree(take, state, rays.state),
        energy=_keep(take, jnp.full_like(rays.energy, launch.quantum), rays.energy),
        alive=rays.alive | take,
    ), launched + jnp.sum(take)


class StepResult(NamedTuple):
    """The result of one cell crossing of one ray.

    Attributes
    ----------
    cell
        The cell that contains the ray after the step.
    state
        The ray after the step.
    energy
        The energy that the ray carries after the step.
    alive
        True if the ray is still in flight.
    cell_deposit
        The energy that the crossed cell absorbs.
    face
        The face id of the exit face.
    face_deposit
        The energy that the exit face absorbs. Zero at an interior face.
    """

    cell: Index
    state: RayState
    energy: Scalar
    alive: Flag
    cell_deposit: Scalar
    face: Index
    face_deposit: Scalar


def step(
    key: KeyArray,
    scene: Scene,
    path: Path[Any],
    interactions: Interactions,
    absorptivity: Float[Array, " ncell"],
    cell: Index,
    state: RayState,
    energy: Scalar,
) -> StepResult:
    """Move one ray across one cell, and deposit energy along its path.

    Parameters
    ----------
    key
        The PRNG key for a boundary interaction.
    scene
        The device arrays of the mesh.
    path
        The ray kinematics of the run.
    interactions
        The interaction table.
    absorptivity
        The absorption coefficient of each cell.
    cell
        The cell that contains the ray.
    state
        The ray.
    energy
        The energy that the ray carries.

    Returns
    -------
    result
        The ray after the step, and the energy it deposited.
    """
    normal, offset = scene.cells.normal[cell], scene.cells.offset[cell]
    slot, parameter = path.crossing(normal, offset, state)
    # A corner or grazing hit can give a small negative parameter.
    parameter = jnp.maximum(parameter, 0.0)

    moved, flown = path.advance(state, parameter)
    with jax.named_scope("attenuate"):
        transmission = jnp.exp(-absorptivity[cell] * flown)
        cell_deposit = energy * (1 - transmission)
        energy = energy * transmission
    state = moved

    face = scene.cells.face[cell, slot]
    neighbour = scene.cells.neighbour[cell, slot]
    at_boundary = neighbour < 0

    group = jnp.where(at_boundary, scene.faces.group[face], 0)
    hit_state, absorbed, terminated = interaction.apply(path, interactions, group, key, state, normal[slot])

    state = _keep_tree(at_boundary, hit_state, state)
    face_deposit = jnp.where(at_boundary, energy * absorbed, 0.0)

    return StepResult(
        cell=jnp.where(at_boundary, cell, neighbour),
        state=state,
        energy=energy - face_deposit,
        alive=~(at_boundary & terminated),
        cell_deposit=cell_deposit,
        face=face,
        face_deposit=face_deposit,
    )


# (step index, key, rays launched so far, the batch, the accumulator)
type Carry = tuple[Index, KeyArray, Index, Rays, Deposit]


@eqx.filter_jit
def march(
    key: KeyArray,
    rays: Rays,
    scene: Scene,
    path: Path[Any],
    interactions: Interactions,
    absorptivity: Float[Array, " ncell"],
    max_steps: int = 1000,
    launch: Launch | None = None,
    extinction: float = 1e-8,
) -> tuple[Rays, Deposit]:
    """March a batch of rays until all stop or the step limit is reached.

    Parameters
    ----------
    key
        The PRNG key for the march.
    rays
        The first batch of rays.
    scene
        The device arrays of the mesh.
    path
        The ray kinematics of the run.
    interactions
        The interaction table.
    absorptivity
        The absorption coefficient of each cell.
    max_steps
        The largest number of steps. It is a safety limit: the march stops
        when all rays stop.
    launch
        The parameters to refill dead lanes, or None to march only the first batch.
    extinction
        The fraction of the largest initial ray energy below which a ray stops
        and deposits its remaining energy in its cell. At zero, only rays with
        no energy stop.

    Returns
    -------
    rays
        The batch at the end of the march. Rays that are still alive hit the step limit.
    deposit
        The energy that the rays deposited, and the energy that escaped.
    """
    floor = extinction * jnp.max(rays.energy)

    def body(carry: Carry) -> Carry:
        index, key, launched, rays, deposit = carry
        key, subkey, refill_key = jax.random.split(key, 3)
        # An extinct ray takes no step: its remainder is deposited where it
        # stands, so the balance is the same one a full march would have reached.
        extinct = rays.alive & (rays.energy <= floor)
        stepping = rays.alive & ~extinct

        crossed = jax.vmap(step, in_axes=(0, None, None, None, None, 0, 0, 0))(
            jax.random.split(subkey, rays.cell.shape[0]),
            scene,
            path,
            interactions,
            absorptivity,
            rays.cell,
            rays.state,
            rays.energy,
        )

        # Whatever a terminating ray still carries left the domain: an absorbing
        # face already took all of it, an escaping one takes none.
        leaving = stepping & ~crossed.alive

        # The scatter *must* be outside the vmap; inside it, XLA materialises a
        # full per-ray copy of the accumulator and memory use explodes.
        with jax.named_scope("scatter"):
            deposit = Deposit(
                cells=deposit.cells.at[rays.cell].add(
                    jnp.where(extinct, rays.energy, jnp.where(stepping, crossed.cell_deposit, 0.0))
                ),
                faces=deposit.faces.at[crossed.face].add(jnp.where(stepping, crossed.face_deposit, 0.0)),
                escaped=deposit.escaped + jnp.sum(jnp.where(leaving, crossed.energy, 0.0)),
            )
        rays = Rays(
            cell=_keep(stepping, crossed.cell, rays.cell),
            state=_keep_tree(stepping, crossed.state, rays.state),
            energy=jnp.where(extinct | leaving, 0.0, _keep(stepping, crossed.energy, rays.energy)),
            alive=stepping & crossed.alive,
        )

        if launch is not None:
            rays, launched = jax.lax.cond(
                (index % launch.interval == 0) & (launched < launch.budget),
                lambda: _refill(refill_key, rays, launched, launch, scene, path),
                lambda: (rays, launched),
            )
        return index + 1, key, launched, rays, deposit

    def cond(carry: Carry) -> Flag:
        index, _, launched, rays, _ = carry
        pending = launched < launch.budget if launch is not None else False
        return (index < max_steps) & (jnp.any(rays.alive) | pending)

    deposit = Deposit(cells=jnp.zeros(scene.num_cells), faces=jnp.zeros(scene.num_faces), escaped=jnp.zeros(()))
    launched = jnp.sum(rays.alive)
    start = jnp.asarray(0)
    _, _, _, rays, deposit = jax.lax.while_loop(cond, body, (start, key, launched, rays, deposit))
    return rays, deposit


@eqx.filter_jit
def launch_and_march(
    key: KeyArray,
    scene: Scene,
    path: Path[Any],
    interactions: Interactions,
    absorptivity: Float[Array, " ncell"],
    sources: Sources,
    num_rays: int,
    lanes: int | None = None,
    max_steps: int = 100_000,
    refill_interval: int = 8,
    extinction: float = 1e-8,
) -> tuple[Rays, Deposit]:
    """Launch rays from the sources and march them until all `num_rays` rays stop.

    Parameters
    ----------
    key
        The PRNG key for the run.
    scene
        The device arrays of the mesh.
    path
        The ray kinematics of the run. It must agree with the geometry of `scene`.
    interactions
        The interaction table.
    absorptivity
        The absorption coefficient of each cell.
    sources
        The emission distribution.
    num_rays
        The total number of rays to launch. Each ray carries `sources.total / num_rays`.
    lanes
        The number of rays in flight at one time, or None for `num_rays`. A
        value larger than `num_rays` becomes `num_rays`.
    max_steps
        The largest number of steps. It is a safety limit: the march stops
        when all rays stop.
    refill_interval
        The number of steps between refills of dead lanes. The best value
        changes with the problem.
    extinction
        The fraction of the ray energy at launch below which a ray stops and
        deposits its remaining energy in its cell.

    Returns
    -------
    rays
        The batch at the end of the march. Rays that are still alive hit the step limit.
    deposit
        The energy that the rays deposited, and the energy that escaped. With the
        energy of the live rays, it sums to the energy of all launched rays.
    """
    # A lane past the ray count has no ray to put in it, and would launch one
    # carrying a quantum the budget never counted.
    lanes = min(num_rays, lanes or num_rays)

    key_launch, key_march = jax.random.split(key)
    quantum = sources.total / num_rays

    state, cell = jax.vmap(sample_source, in_axes=(0, None, None, None))(
        jax.random.split(key_launch, lanes), sources, scene, path
    )
    rays = Rays(cell=cell, state=state, energy=jnp.full(lanes, quantum), alive=jnp.ones(lanes, dtype=bool))

    launch = None
    if lanes < num_rays:
        launch = Launch(sources=sources, budget=jnp.asarray(num_rays), quantum=quantum, interval=refill_interval)
    return march(
        key_march,
        rays,
        scene,
        path,
        interactions,
        absorptivity,
        max_steps=max_steps,
        launch=launch,
        extinction=extinction,
    )


# The entry point: the same march, with a `lanes` that sizes itself to the
# device when left out. A lane count passed explicitly is marched as given.
trace = tuning.autosized("lanes")(launch_and_march)
