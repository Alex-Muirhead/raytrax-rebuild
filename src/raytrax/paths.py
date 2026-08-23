import abc
from typing import ClassVar, override

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from equinox._misc import default_floating_dtype
from jaxtyping import Array, Float

from raytrax.random import cosine_hemisphere, simplex, sphere
from raytrax.shapes import Index, KeyArray, Normals, Offsets, Scalar, Vector

# Meridional column order for a revolved mesh.
AXIAL, RADIAL = 0, 1


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


class HyperbolicState(RayState):
    """Adds the swirl `r |p x e|`, conserved along the ray and through a bounce."""

    swirl: Float[Array, "..."]


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


class AxisymmetricPath(Path[HyperbolicState]):
    """Revolved rays: a 2D mesh read as the meridional half-plane `(x, r)`.

    A straight 3D ray projects to a hyperbola. The crossing parameter is a step
    in the hyperbola's own parameter, not a distance; `advance` converts it.
    """

    @override
    def launch(self, position: Vector, direction: Float[Array, " 3"], travel: Scalar) -> HyperbolicState:
        # The swirl is the azimuthal component times the radius. Reading it off
        # the 3D draw keeps full precision for a ray that nearly meets the axis,
        # where recovering it from `1 - |tangent|^2` would cancel away.
        swirl = position[RADIAL] * jnp.abs(direction[2])
        return HyperbolicState(position, direction[:2], travel, swirl)

    def _frame(self, state: HyperbolicState) -> tuple[Vector, Vector, Scalar, Scalar]:
        """Asymptote directions and their coefficients, anchored at the ray.

        The parametrisation is fixed so the ray sits at `w = 1`; a frame pinned
        to the hyperbola's centre instead loses the position to cancellation
        whenever that centre is far away.
        """
        axial, radial = state.tangent[AXIAL], state.tangent[RADIAL]
        # A sum, where `1 - axial^2` would cancel for a near-axial ray.
        reduced = radial**2 + (state.swirl / state.position[RADIAL]) ** 2
        rate = jnp.sqrt(reduced)

        scale = (state.swirl / reduced) ** 2
        near = state.position[RADIAL] * (rate + jnp.abs(radial)) / reduced
        far = scale / near
        # Pair them: the sum never cancels and the product is exact, so the
        # small coefficient keeps full relative precision.
        first, second = jnp.where(radial >= 0, near, far), jnp.where(radial >= 0, far, near)
        return jnp.stack([axial, rate]), jnp.stack([-axial, rate]), first, second

    @override
    def crossing(self, normal: Normals, offset: Offsets, state: HyperbolicState) -> tuple[Index, Scalar]:
        forward, backward, first, second = self._frame(state)

        along = first * (normal @ forward)
        against = second * (normal @ backward)
        distance = normal @ state.position - offset

        # (w - 1) solves this, so the ray's own position anchors the quadratic.
        linear = 2 * distance + along - against
        discriminant = linear**2 - 8 * distance * along
        root = jnp.sqrt(jnp.where(discriminant > 0, discriminant, jnp.nan))
        pair = -(linear + jnp.where(linear >= 0, 1.0, -1.0) * root) / 2

        step = jnp.stack([pair / along, 2 * distance / pair])
        # Outgoing: the sign of dg/dw. A convex trace can leave through the face
        # it entered by, so the derivative and not the ordering picks the root.
        outgoing = (1 + step) ** 2 * along > against
        step = jnp.where((step > 0) & outgoing & jnp.isfinite(step), step, jnp.nan)

        best = jnp.nanmin(step, axis=0)
        slot = jnp.nanargmin(best)
        return slot, best[slot]

    @override
    def advance(self, state: HyperbolicState, parameter: Scalar) -> HyperbolicState:
        forward, backward, first, second = self._frame(state)
        angle = 1 + parameter

        moved = parameter / 2 * (first * forward - second * backward / angle)
        flown = ((first * angle - second / angle) - (first - second)) / 2
        rate = forward[RADIAL]
        radial = rate * (first - second / angle**2) / (first + second / angle**2)

        return HyperbolicState(
            position=state.position + moved,
            # The axial component is untouched by the march; only a wall moves it.
            tangent=jnp.stack([state.tangent[AXIAL], radial]),
            travel=state.travel + flown,
            swirl=state.swirl,
        )
