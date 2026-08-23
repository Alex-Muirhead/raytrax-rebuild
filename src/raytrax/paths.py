import abc
from typing import ClassVar, override

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from equinox._misc import default_floating_dtype
from jaxtyping import Array, Float

from raytrax.random import cosine_hemisphere, sphere
from raytrax.shapes import Index, KeyArray, Normals, Offsets, Scalar, Vector


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

    # Instance methods, not classmethods: they read only `spatial_dim` today, but
    # `AxisymmetricPath` carries a static field for which in-plane coordinate is
    # axial, and reducing the draw has to respect it.
    def isotropic(self, key: KeyArray, ndim: int) -> Vector:
        """A direction uniform on the sphere, reduced to the modelled plane."""
        return sphere(key, self.spatial_dim)[:ndim]

    def lambertian(self, key: KeyArray, normal: Vector) -> Vector:
        """Cosine-weighted about an in-plane normal, embedded then reduced."""
        ndim = normal.shape[-1]
        return cosine_hemisphere(key, jnp.pad(normal, (0, self.spatial_dim - ndim)))[:ndim]

    @abc.abstractmethod
    def crossing(self, normal: Normals, offset: Offsets, state: RayState) -> tuple[Index, Scalar]:
        """The slot the ray leaves through, and the parameter at which it does."""

    @abc.abstractmethod
    def advance(self, state: RayState, parameter: Scalar) -> RayState:
        """Step the ray forward by `parameter`, accumulating true 3D length."""

    @abc.abstractmethod
    def reflect(self, state: RayState, normal: Vector) -> RayState:
        """Mirror the tangent about an outward unit normal."""


class LinearPath(Path):
    """Straight rays: through a 3D mesh, or through a 2D one read as extruded.

    Both are the same solve. A ray of the extruded domain projects to a straight
    line at speed `|tangent|`, so the parameter the crossing returns is already
    the 3D distance flown.
    """

    @override
    def crossing(self, normal: Normals, offset: Offsets, state: RayState) -> tuple[Index, Scalar]:
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

    @override
    def advance(self, state: RayState, parameter: Scalar) -> RayState:
        return RayState(
            position=state.position + parameter * state.tangent,
            tangent=state.tangent,
            travel=state.travel + parameter,
        )

    @override
    def reflect(self, state: RayState, normal: Vector) -> RayState:
        # Outward unit normal, so the mirror preserves `|tangent|` -- including
        # the out-of-plane content it stands for, which an extruded or revolved
        # wall cannot touch.
        return RayState(
            position=state.position,
            tangent=state.tangent - 2 * (state.tangent @ normal) * normal,
            travel=state.travel,
        )
