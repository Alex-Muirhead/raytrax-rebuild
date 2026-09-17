import functools
from typing import Any, NamedTuple

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Bool, Float, Int

from raytrax import interactions as interaction
from raytrax import jit_tools
from raytrax.paths import RayState
from raytrax.problem import Problem
from raytrax.scene import Scene
from raytrax.shapes import Flag, Index, KeyArray, Scalar
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

    @classmethod
    def zeros(cls, scene: Scene) -> "Deposit":
        """Make an empty deposit for a mesh.

        Parameters
        ----------
        scene
            The device arrays of the mesh.

        Returns
        -------
        deposit
            The deposit with no energy in it.
        """
        return cls(cells=jnp.zeros(scene.num_cells), faces=jnp.zeros(scene.num_faces), escaped=jnp.zeros(()))


class Rays(eqx.Module):
    """A batch of rays in the mesh, one for each lane.

    Attributes
    ----------
    cell
        The cell that contains each ray.
    state
        The position and direction of each ray.
    energy
        The energy that each ray carries, as a fraction of its launch energy.
    alive
        True for each lane that holds a ray in flight.
    """

    cell: Int[Array, " nray"]
    state: RayState
    energy: Float[Array, " nray"]
    alive: Bool[Array, " nray"]


def launch(key: KeyArray, problem: Problem, lanes: int) -> Rays:
    """Draw a batch of new rays from the sources, each with energy 1.

    Parameters
    ----------
    key
        The PRNG key for the new rays.
    problem
        The problem whose sources emit the rays.
    lanes
        The number of rays in the batch.

    Returns
    -------
    rays
        The batch, with all rays alive.
    """
    state, cell = jax.vmap(sample_source, in_axes=(0, None, None, None))(
        jax.random.split(key, lanes), problem.sources, problem.scene, problem.path
    )
    return Rays(cell=cell, state=state, energy=jnp.ones(lanes), alive=jnp.ones(lanes, dtype=bool))


@jax.named_scope("refill")
def refill(key: KeyArray, rays: Rays, launched: Index, budget: Index, problem: Problem) -> tuple[Rays, Index]:
    """Put new rays in dead lanes, up to the launch budget.

    Parameters
    ----------
    key
        The PRNG key for the new rays.
    rays
        The batch of rays.
    launched
        The number of rays launched before this refill.
    budget
        The number of rays launched when the run is complete.
    problem
        The problem whose sources emit the new rays.

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
    take = dead & (launched + rank < budget)

    fresh = launch(key, problem, rays.cell.shape[0])
    return _keep_tree(take, fresh, rays), launched + jnp.sum(take)


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


def step(key: KeyArray, problem: Problem, cell: Index, state: RayState, energy: Scalar) -> StepResult:
    """Move one ray across one cell, and deposit energy along its path.

    Parameters
    ----------
    key
        The PRNG key for a boundary interaction.
    problem
        The problem that the ray moves through.
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
    scene, path = problem.scene, problem.path
    normal, offset = scene.cells.normal[cell], scene.cells.offset[cell]
    slot, parameter = path.crossing(normal, offset, state)
    # A corner or grazing hit can give a small negative parameter.
    parameter = jnp.maximum(parameter, 0.0)

    moved, flown = path.advance(state, parameter)
    with jax.named_scope("attenuate"):
        transmission = jnp.exp(-problem.absorptivity[cell] * flown)
        cell_deposit = energy * (1 - transmission)
        energy = energy * transmission
    state = moved

    face = scene.cells.face[cell, slot]
    neighbour = scene.cells.neighbour[cell, slot]
    at_boundary = neighbour < 0

    group = jnp.where(at_boundary, scene.faces.group[face], 0)
    hit_state, absorbed, terminated = interaction.apply(path, problem.interactions, group, key, state, normal[slot])

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


@eqx.filter_jit
def march(key: KeyArray, rays: Rays, problem: Problem, steps: int, floor: float | Scalar = 0.0) -> tuple[Rays, Deposit]:
    """Move each live ray across a fixed number of cells, and deposit energy along its path.

    Parameters
    ----------
    key
        The PRNG key for the march.
    rays
        The batch of rays.
    problem
        The problem that the rays move through.
    steps
        The number of steps. The march takes all of them, also when all rays stop earlier.
    floor
        The ray energy at or below which a ray stops and deposits its remaining
        energy in its cell. At zero, only rays with no energy stop.

    Returns
    -------
    rays
        The batch after the last step.
    deposit
        The energy that the rays deposited in these steps, and the energy that escaped.
    """

    def body(carry: tuple[Rays, Deposit], key: KeyArray) -> tuple[tuple[Rays, Deposit], None]:
        rays, deposit = carry
        # An extinct ray takes no step: its remainder is deposited where it
        # stands, so the balance is the same one a full march would have reached.
        extinct = rays.alive & (rays.energy <= floor)
        stepping = rays.alive & ~extinct

        crossed = jax.vmap(step, in_axes=(0, None, 0, 0, 0))(
            jax.random.split(key, rays.cell.shape[0]), problem, rays.cell, rays.state, rays.energy
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
        return (rays, deposit), None

    (rays, deposit), _ = jax.lax.scan(body, (rays, Deposit.zeros(problem.scene)), jax.random.split(key, steps))
    return rays, deposit


# (steps taken, key, rays launched so far, the batch, the accumulator)
type Carry = tuple[Index, KeyArray, Index, Rays, Deposit]


class Tally(eqx.Module):
    """The running sums of a Monte-Carlo run on one problem, which a later run can extend.

    Attributes
    ----------
    key
        The PRNG key for the next run.
    launched
        The number of rays launched so far.
    rays
        The lanes, with the rays that are still in flight.
    deposit
        The summed energy that the rays deposited and that escaped, in units of
        the launch energy of one ray.
    """

    key: KeyArray
    launched: Int[Array, ""]
    rays: Rays
    deposit: Deposit

    @classmethod
    def empty(cls, key: KeyArray, problem: Problem, lanes: int) -> "Tally":
        """Make a tally with no rays launched.

        Parameters
        ----------
        key
            The PRNG key for the first run.
        problem
            The problem to simulate.
        lanes
            The number of rays in flight at one time, for this tally and every run that extends it.

        Returns
        -------
        tally
            The tally, with all lanes empty.
        """
        shapes = jax.eval_shape(functools.partial(launch, lanes=lanes), key, problem)

        def zeros(leaf: Any) -> Array:
            return jnp.zeros(leaf.shape, leaf.dtype)

        rays = jax.tree.map(zeros, shapes)
        return cls(key=key, launched=jnp.asarray(0), rays=rays, deposit=Deposit.zeros(problem.scene))

    def estimate(self, problem: Problem) -> Deposit:
        """Give the deposit and escape of the problem, in the power units of its sources.

        Parameters
        ----------
        problem
            The problem that the tally was run on.

        Returns
        -------
        deposit
            The mean energy per launched ray, times the total emitted power.
        """
        scale = problem.sources.total / self.launched
        return Deposit(
            cells=self.deposit.cells * scale, faces=self.deposit.faces * scale, escaped=self.deposit.escaped * scale
        )

    def in_flight(self, problem: Problem) -> Scalar:
        """Give the power that is still in flight at the step limit, in the units of `estimate`.

        Parameters
        ----------
        problem
            The problem that the tally was run on.

        Returns
        -------
        power
            The power that the live rays carry.
        """
        return jnp.sum(self.rays.energy) * problem.sources.total / self.launched

    @jit_tools.jit_dynamic("num_rays")
    def extend(
        self,
        problem: Problem,
        num_rays: int | Index,
        max_steps: int = 100_000,
        refill_interval: int = 8,
        extinction: float = 1e-8,
    ) -> "Tally":
        """Launch `num_rays` more rays into the tally, and march until they all stop.

        The problem must be the one that the tally was run on. Nothing checks this.

        Parameters
        ----------
        problem
            The problem that the tally was run on.
        num_rays
            The number of rays to add. Zero only finishes the rays still in flight.
        max_steps
            The largest number of steps in this run, rounded up to a multiple of
            `refill_interval`. It is a safety limit: the run stops when all rays stop.
        refill_interval
            The number of steps between refills of dead lanes, and between checks
            for the end of the run. The best value changes with the problem.
        extinction
            The fraction of the launch energy of a ray at or below which the ray
            stops and deposits its remaining energy in its cell.

        Returns
        -------
        tally
            The tally with the new rays added. Rays that are still alive hit the step limit.
        """
        budget = self.launched + num_rays

        def body(carry: Carry) -> Carry:
            taken, key, launched, rays, deposit = carry
            key, key_refill, key_march = jax.random.split(key, 3)
            rays, launched = jax.lax.cond(
                launched < budget,
                lambda: refill(key_refill, rays, launched, budget, problem),
                lambda: (rays, launched),
            )
            rays, chunk = march(key_march, rays, problem, refill_interval, extinction)
            return taken + refill_interval, key, launched, rays, jax.tree.map(jnp.add, deposit, chunk)

        def cond(carry: Carry) -> Flag:
            taken, _, launched, rays, _ = carry
            return (taken < max_steps) & (jnp.any(rays.alive) | (launched < budget))

        start: Carry = (jnp.asarray(0), self.key, self.launched, self.rays, self.deposit)
        _, key, launched, rays, deposit = jax.lax.while_loop(cond, body, start)
        return Tally(key=key, launched=launched, rays=rays, deposit=deposit)


def launch_and_march(
    key: KeyArray,
    problem: Problem,
    num_rays: int,
    lanes: int | None = None,
    max_steps: int = 100_000,
    refill_interval: int = 8,
    extinction: float = 1e-8,
) -> Tally:
    """Start a tally on a problem, and run `num_rays` rays into it.

    Parameters
    ----------
    key
        The PRNG key for the tally.
    problem
        The problem to simulate.
    num_rays
        The number of rays to launch.
    lanes
        The number of rays in flight at one time, or None for `num_rays`.
    max_steps
        The largest number of steps, as in `Tally.extend`.
    refill_interval
        The number of steps between refills of dead lanes, as in `Tally.extend`.
    extinction
        The fraction of the launch energy at or below which a ray stops, as in `Tally.extend`.

    Returns
    -------
    tally
        The tally. `Tally.estimate` gives the deposit, and `Tally.extend` adds more rays.
    """
    tally = Tally.empty(key, problem, lanes or num_rays)
    return tally.extend(problem, num_rays, max_steps, refill_interval, extinction)


# The entry point: the same run, with a `lanes` that sizes itself to the
# device when left out. A lane count passed explicitly is marched as given.
trace = jit_tools.autosized("lanes")(launch_and_march)
