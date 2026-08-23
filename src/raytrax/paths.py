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

    `tangent` is unit in 3D but shorter in 2D, where the deficit stands for the
    out-of-plane component that is not stored.
    """

    position: Float[Array, "... ndim"]
    tangent: Float[Array, "... ndim"]
    travel: Float[Array, "..."]


class Path(eqx.Module):
    """Ray kinematics: how a ray crosses a cell, and how far that really is.

    Directions are drawn in 3D and reduced to the modelled plane.
    """

    spatial_dim: ClassVar[int] = 3

    # Not classmethods: `AxisymmetricPath` reduces the draw against a static field.
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

    The crossing parameter is already the 3D distance flown.
    """

    @override
    def crossing(self, normal: Normals, offset: Offsets, state: RayState) -> tuple[Index, Scalar]:
        epsilon = np.finfo(default_floating_dtype()).resolution

        distance = normal @ state.position - offset
        alignment = normal @ state.tangent
        travel = -distance / alignment

        # Keep only faces the ray heads out through. Scaling the threshold by
        # `|tangent|` makes the rejection independent of how much of the ray
        # points out of plane; padded slots have a zero normal and drop out here.
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
        # Mirroring preserves `|tangent|`, and so the out-of-plane content it
        # stands for, which an extruded or revolved wall cannot touch.
        return RayState(
            position=state.position,
            tangent=state.tangent - 2 * (state.tangent @ normal) * normal,
            travel=state.travel,
        )
