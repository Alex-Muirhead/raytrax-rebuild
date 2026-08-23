import abc
from typing import ClassVar, override

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from equinox._misc import default_floating_dtype
from jaxtyping import Array, Float

from raytrax.random import cosine_hemisphere, simplex, sphere
from raytrax.shapes import Index, KeyArray, Normals, Offsets, Scalar, Vector


class RayState(eqx.Module):
    """A half-line from `position` along `tangent`, having already flown `travel`.

    `tangent` is unit in 3D but shorter in 2D, where the deficit stands for the
    out-of-plane component that is not stored.
    """

    position: Float[Array, "... ndim"]
    tangent: Float[Array, "... ndim"]
    travel: Float[Array, "..."]


class LinearState(RayState):
    """Adds the tangent's magnitude, fixed for the life of the ray."""

    speed: Float[Array, "..."]


class Path[S: RayState](eqx.Module):
    """Ray kinematics: how a ray crosses a cell, and how far that really is.

    Directions are drawn in 3D and reduced by `launch`, which is also where a
    mode picks up whatever it conserves.
    """

    spatial_dim: ClassVar[int] = 3

    # Not classmethods: a mode may reduce the draw against a static field.
    def isotropic(self, key: KeyArray) -> Float[Array, " 3"]:
        """A 3D direction uniform on the sphere."""
        return sphere(key, self.spatial_dim)

    def lambertian(self, key: KeyArray, normal: Vector) -> Float[Array, " 3"]:
        """A 3D cosine-weighted direction about an in-plane normal."""
        return cosine_hemisphere(key, jnp.pad(normal, (0, self.spatial_dim - normal.shape[-1])))

    def simplex_point(self, key: KeyArray, corners: Float[Array, "ncorner ndim"]) -> Vector:
        """A point uniform over the simplex spanned by `corners`."""
        return simplex(key, corners.shape[0] - 1) @ corners

    @abc.abstractmethod
    def launch(self, position: Vector, direction: Float[Array, " 3"], travel: Scalar) -> S:
        """Reduce a 3D direction into a ray state at `position`."""

    def reflect(self, state: S, normal: Vector) -> S:
        """Mirror the tangent about an outward unit normal."""
        # A wall leaves the radius alone and preserves `|tangent|`, so both the
        # speed and the swirl survive untouched; only the tangent moves.
        tangent = state.tangent - 2 * (state.tangent @ normal) * normal
        return eqx.tree_at(lambda leaf: leaf.tangent, state, tangent)

    @abc.abstractmethod
    def crossing(self, normal: Normals, offset: Offsets, state: S) -> tuple[Index, Scalar]:
        """The slot the ray leaves through, and the parameter at which it does."""

    @abc.abstractmethod
    def advance(self, state: S, parameter: Scalar) -> S:
        """Step the ray forward by `parameter`, accumulating true 3D length."""


class LinearPath(Path[LinearState]):
    """Straight rays: through a 3D mesh, or through a 2D one read as extruded.

    The crossing parameter is already the 3D distance flown.
    """

    @override
    def launch(self, position: Vector, direction: Float[Array, " 3"], travel: Scalar) -> LinearState:
        tangent = direction[: position.shape[-1]]
        return LinearState(position, tangent, travel, jnp.linalg.vector_norm(tangent))

    @override
    def crossing(self, normal: Normals, offset: Offsets, state: LinearState) -> tuple[Index, Scalar]:
        epsilon = np.finfo(default_floating_dtype()).resolution

        distance = normal @ state.position - offset
        alignment = normal @ state.tangent
        travel = -distance / alignment

        # Keep only faces the ray heads out through. Scaling the threshold by
        # the speed makes the rejection independent of how much of the ray
        # points out of plane; padded slots have a zero normal and drop out here.
        travel = jnp.where(alignment > epsilon * state.speed, travel, jnp.nan)
        slot = jnp.nanargmin(travel)
        return slot, travel[slot]

    @override
    def advance(self, state: LinearState, parameter: Scalar) -> LinearState:
        return LinearState(
            position=state.position + parameter * state.tangent,
            tangent=state.tangent,
            travel=state.travel + parameter,
            speed=state.speed,
        )
