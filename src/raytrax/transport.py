from typing import Any, NamedTuple

import equinox as eqx
import jax
import jax.numpy as jnp
from jax.errors import JaxRuntimeError
from jaxtyping import Array, Bool, Float, Int

from raytrax import interactions as interaction
from raytrax.interactions import Interactions
from raytrax.paths import Path, RayState
from raytrax.scene import Scene
from raytrax.shapes import Flag, Index, KeyArray, Scalar
from raytrax.sources import Sources
from raytrax.sources import sample as sample_source

# Narrower than this a batch is not worth marching, so an estimate that has been
# halved down to it is out of memory for some reason other than its width.
MIN_LANES = 1 << 12


def _keep(mask: Bool[Array, "..."], new: Array, old: Array) -> Array:
    """Take `new` where `mask`, else `old`, broadcasting the mask up to `old`."""
    return jnp.where(mask.reshape(mask.shape + (1,) * (old.ndim - mask.ndim)), new, old)


def _keep_tree[T](mask: Bool[Array, "..."], new: T, old: T) -> T:
    """`_keep` leafwise over two pytrees of matching structure."""

    def keep(new_leaf: Array, old_leaf: Array) -> Array:
        return _keep(mask, new_leaf, old_leaf)

    return jax.tree.map(keep, new, old)


class Deposit(eqx.Module):
    """Where the energy ended up.

    `escaped` is banked as a ray leaves rather than left in the lane it dies
    in, which refilling would overwrite.
    """

    cells: Float[Array, " ncell"]
    faces: Float[Array, " nface"]
    escaped: Scalar


class Launch(eqx.Module):
    """Refill dead lanes from a source until a launch budget is spent.

    Every launched ray carries exactly `quantum`, so the deposited total is
    independent of how launches land across the lanes.
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


def _refill(
    key: KeyArray, rays: Rays, launched: Index, launch: Launch, scene: Scene, path: Path[Any]
) -> tuple[Rays, Index]:
    """Overwrite dead lanes with fresh rays, up to the remaining budget."""
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
    """One cell crossing: where the ray went, and what it left behind."""

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
    """Cross one cell, depositing continuously along the way."""
    normal, offset = scene.cells.normal[cell], scene.cells.offset[cell]
    slot, parameter = path.crossing(normal, offset, state)
    # A corner or grazing hit can give a small negative parameter.
    parameter = jnp.maximum(parameter, 0.0)

    moved, flown = path.advance(state, parameter)
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
    """Walk rays until they terminate or `max_steps` runs out.

    `max_steps` is a safety net, not a cost: the batch pays for its longest ray.
    A step costs O(lanes) rather than O(alive), so pass a `Launch` to refill
    dead lanes and keep the batch full.

    `extinction` is the share of the launch quantum below which a ray stops,
    leaving its remainder in the cell it is in. Zero reaps only exact zeros.
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
def _launch_and_march(
    key: KeyArray,
    scene: Scene,
    path: Path[Any],
    interactions: Interactions,
    absorptivity: Float[Array, " ncell"],
    sources: Sources,
    num_rays: int,
    lanes: int,
    max_steps: int,
    refill_interval: int,
    extinction: float,
) -> tuple[Rays, Deposit]:
    """Fill every lane from the source, then march until `num_rays` are spent."""
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


def _resident_bytes(lanes: int, **traced: Any) -> int:
    """Device bytes one `_launch_and_march` over `lanes` lanes holds at its peak."""
    # pyrefly: ignore[missing-attribute] - it does not model what filter_jit returns
    lowered = _launch_and_march.lower(lanes=lanes, **traced)
    # equinox wraps the jax Compiled; the analyses live on the inner one.
    compiled = lowered.compile().compiled
    analysis = compiled.memory_analysis()
    return analysis.temp_size_in_bytes + analysis.argument_size_in_bytes + analysis.output_size_in_bytes


def lane_capacity(
    key: KeyArray,
    scene: Scene,
    path: Path[Any],
    interactions: Interactions,
    absorptivity: Float[Array, " ncell"],
    sources: Sources,
    max_steps: int = 100_000,
    refill_interval: int = 8,
    extinction: float = 1e-8,
    fraction: float = 0.7,
    probe: tuple[int, int] = (1 << 16, 1 << 18),
) -> int | None:
    """The most lanes this march fits in device memory, or None if the device does not report it.

    `fraction` is the share of the pool to fill. The pool XLA reports is a share
    of the whole card rather than of what is free on it, so the default leaves
    room for a display or a second process; raise it on a dedicated device.
    `probe` is the pair of lane counts compiled to price a lane, and wants to
    sit above the few MiB a march costs whatever the lane count.
    """
    stats = jax.local_devices()[0].memory_stats()
    if not stats or stats.get("bytes_limit") is None:
        return None
    budget = fraction * (stats["bytes_limit"] - stats["bytes_in_use"])

    small, large = probe
    traced = {
        "key": key,
        "scene": scene,
        "path": path,
        "interactions": interactions,
        "absorptivity": absorptivity,
        "sources": sources,
        # Any budget past the probe compiles the refill in, which an automatic
        # lane count always ends up using.
        "num_rays": 2 * large,
        "max_steps": max_steps,
        "refill_interval": refill_interval,
        "extinction": extinction,
    }
    cheap, dear = _resident_bytes(small, **traced), _resident_bytes(large, **traced)
    per_lane = (dear - cheap) / (large - small)
    fixed = cheap - per_lane * small
    return max(MIN_LANES, int((budget - fixed) / per_lane))


def trace(
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
    """Trace `num_rays` rays through `lanes` lanes, refilling as they terminate.

    `lanes` trades memory against occupancy, defaulting to whatever
    `lane_capacity` says the device holds; an estimate that still runs out of
    memory is halved and marched again. Pass it explicitly to keep a sweep from
    paying the estimate's two compiles per call. `refill_interval` is worth
    sweeping on a new problem; sampling costs enough that refilling every step
    is a loss. `extinction` is the share of a ray's quantum below which it
    stops, and is what bounds the tail.
    """
    if lanes is None:
        lanes = lane_capacity(
            key,
            scene,
            path,
            interactions,
            absorptivity,
            sources,
            max_steps=max_steps,
            refill_interval=refill_interval,
            extinction=extinction,
        )
    lanes = min(num_rays, lanes or num_rays)

    while True:
        try:
            return _launch_and_march(
                key,
                scene,
                path,
                interactions,
                absorptivity,
                sources,
                num_rays=num_rays,
                lanes=lanes,
                max_steps=max_steps,
                refill_interval=refill_interval,
                extinction=extinction,
            )
        except JaxRuntimeError as error:
            # XLA reports a share of the whole card, so an estimate can still
            # meet a display or another process. The answer does not depend on
            # the lane count, so a narrower batch is the same march.
            if "RESOURCE_EXHAUSTED" not in str(error) or lanes <= MIN_LANES:
                raise
            lanes //= 2
