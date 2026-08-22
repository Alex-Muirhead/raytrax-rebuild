import equinox as eqx
import jax.numpy as jnp
import numpy as np
from equinox._misc import default_floating_dtype
from jaxtyping import Array, Float, Int


class RayState(eqx.Module):
    """A half-line from `position` along `tangent`, having already flown `travel`.

    `position` advances to each crossing point as the ray is walked, so the
    intersection stays local to the current cell and never loses precision to a
    distant origin. `travel` accumulates physical path length and the geometry
    never reads it back.
    """

    position: Float[Array, "... ndim"]
    tangent: Float[Array, "... ndim"]
    travel: Float[Array, "..."]


class Path(eqx.Module):
    """Ray kinematics: how a ray crosses a cell, and how far that really is.

    Splitting this out is what keeps the 2D modes additive. A planar simulation
    corrects `distance` for the unmodelled third dimension; an axisymmetric one
    replaces the linear solve with a hyperbolic one. Neither touches transport.
    """

    def crossing(self, normal, offset, state: RayState) -> tuple[Int[Array, "..."], Float[Array, "..."]]:
        raise NotImplementedError

    def advance(self, state: RayState, parameter) -> RayState:
        raise NotImplementedError

    def reflect(self, state: RayState, normal) -> RayState:
        raise NotImplementedError


class LinearPath(Path):
    """Straight rays through a 3D mesh, or a 2D slice taken at face value."""

    def crossing(self, normal, offset, state):
        epsilon = np.finfo(default_floating_dtype()).resolution

        # Signed distance "outside" each half-space.
        distance = normal @ state.position - offset
        alignment = normal @ state.tangent
        travel = -distance / alignment

        # Only faces the ray is actually heading out through; a null half-space
        # has zero alignment, so padded slots drop out here.
        travel = jnp.where(alignment > epsilon, travel, jnp.nan)
        slot = jnp.nanargmin(travel)
        return slot, travel[slot]

    def advance(self, state, parameter):
        return RayState(
            position=state.position + parameter * state.tangent,
            tangent=state.tangent,
            travel=state.travel + parameter,
        )

    def reflect(self, state, normal):
        # Outward unit normal, so the mirror preserves length.
        return RayState(
            position=state.position,
            tangent=state.tangent - 2 * (state.tangent @ normal) * normal,
            travel=state.travel,
        )
