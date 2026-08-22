from typing import ClassVar

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from equinox._misc import default_floating_dtype
from jaxtyping import Array, Float, Int

from raytrax.random import cosine_hemisphere, sphere


class RayState(eqx.Module):
    """A half-line from `position` along `tangent`, having already flown `travel`.

    `position` advances to each crossing point as the ray is walked, so the
    intersection stays local to the current cell and never loses precision to a
    distant origin. `travel` accumulates physical path length and the geometry
    never reads it back.

    `tangent` is the in-plane velocity of a ray parametrised by its *3D* arc
    length, so `|tangent| <= 1` and the deficit is whatever points out of the
    modelled plane. In 3D nothing is missing and it is a unit vector. In 2D the
    out-of-plane component is not stored, because the domain's symmetry makes it
    unobservable -- and carrying it as a deficit instead is what lets `travel`
    accumulate the true 3D length with no correction factor.
    """

    position: Float[Array, "... ndim"]
    tangent: Float[Array, "... ndim"]
    travel: Float[Array, "..."]


class Path(eqx.Module):
    """Ray kinematics: how a ray crosses a cell, and how far that really is.

    Splitting this out is what keeps the 2D modes additive. A planar simulation
    reads a 2D mesh as a domain extruded out of plane; an axisymmetric one
    revolves it and replaces the linear solve with a hyperbolic one. Neither
    touches transport.

    Directions are always drawn in 3D and reduced to the modelled plane. That
    reduction is shared by every mode -- axisymmetric draws in the local frame
    at the ray's own radius, where it is the same operation -- so it lives here
    rather than in `sources`.
    """

    spatial_dim: ClassVar[int] = 3

    def isotropic(self, key, ndim: int) -> Float[Array, " ndim"]:
        """A direction uniform on the sphere, reduced to the modelled plane."""
        return sphere(key, self.spatial_dim)[:ndim]

    def lambertian(self, key, normal) -> Float[Array, " ndim"]:
        """Cosine-weighted about an in-plane normal, embedded then reduced."""
        ndim = normal.shape[-1]
        return cosine_hemisphere(key, jnp.pad(normal, (0, self.spatial_dim - ndim)))[:ndim]

    def crossing(self, normal, offset, state: RayState) -> tuple[Int[Array, "..."], Float[Array, "..."]]:
        raise NotImplementedError

    def advance(self, state: RayState, parameter) -> RayState:
        raise NotImplementedError

    def reflect(self, state: RayState, normal) -> RayState:
        raise NotImplementedError


class LinearPath(Path):
    """Straight rays: through a 3D mesh, or through a 2D one read as extruded.

    Both are the same solve. A ray of the extruded domain projects to a straight
    line at speed `|tangent|`, so the parameter the crossing returns is already
    the 3D distance flown.
    """

    def crossing(self, normal, offset, state):
        epsilon = np.finfo(default_floating_dtype()).resolution

        # Signed distance "outside" each half-space.
        distance = normal @ state.position - offset
        alignment = normal @ state.tangent
        travel = -distance / alignment

        # Only faces the ray is actually heading out through. The threshold is
        # relative to `|tangent|`, so which faces it rejects does not depend on
        # how much of the ray points out of plane; a null half-space has a zero
        # normal, so padded slots drop out here whatever the scale. A ray flying
        # exactly out of plane leaves every slot rejected, but that is a measure
        # zero draw.
        scale = jnp.linalg.vector_norm(state.tangent)
        travel = jnp.where(alignment > epsilon * scale, travel, jnp.nan)
        slot = jnp.nanargmin(travel)
        return slot, travel[slot]

    def advance(self, state, parameter):
        return RayState(
            position=state.position + parameter * state.tangent,
            tangent=state.tangent,
            travel=state.travel + parameter,
        )

    def reflect(self, state, normal):
        # Outward unit normal, so the mirror preserves `|tangent|` -- including
        # the out-of-plane content it stands for, which an extruded or revolved
        # wall cannot touch.
        return RayState(
            position=state.position,
            tangent=state.tangent - 2 * (state.tangent @ normal) * normal,
            travel=state.travel,
        )
