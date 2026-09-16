import abc
from typing import override

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from equinox._misc import default_floating_dtype
from jaxtyping import Array, Float

from raytrax.random import cosine_hemisphere, simplex, sphere
from raytrax.shapes import Index, KeyArray, Normals, Offsets, Scalar, Vector

# Column order for a revolved mesh.
AXIAL, RADIAL = 0, 1

# Directions are always drawn in 3D, whatever the mesh dimension.
SPATIAL_DIM = 3


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
    """Adds the angular momentum about the axis, conserved along the ray and through a bounce."""

    # Specific, so a radius times an azimuthal speed rather than a true moment.
    angular_momentum: Float[Array, "..."]


class Path[S: RayState](eqx.Module):
    """Ray kinematics: how a ray crosses a cell, and how far that really is.

    Directions are drawn in 3D and reduced by `launch`, which is also where a
    mode picks up whatever it conserves.
    """

    # Classmethods: a mode holds no state, and anything that varies between
    # modes is class data. Dispatch is still on the instance's type.
    @classmethod
    def isotropic(cls, key: KeyArray) -> Float[Array, " 3"]:
        """A 3D direction uniform on the sphere."""
        return sphere(key, SPATIAL_DIM)

    @classmethod
    def lambertian(cls, key: KeyArray, normal: Vector) -> Float[Array, " 3"]:
        """A 3D cosine-weighted direction about an in-plane normal."""
        return cosine_hemisphere(key, jnp.pad(normal, (0, SPATIAL_DIM - normal.shape[-1])))

    @classmethod
    def simplex_point(cls, key: KeyArray, corners: Float[Array, "ncorner ndim"]) -> Vector:
        """A point uniform over the simplex spanned by `corners`."""
        return simplex(key, corners.shape[0] - 1) @ corners

    @classmethod
    @abc.abstractmethod
    def launch(cls, position: Vector, direction: Float[Array, " 3"], travel: Scalar) -> S:
        """Reduce a 3D direction into a ray state at `position`."""

    @classmethod
    def reflect(cls, state: S, normal: Vector) -> S:
        """Mirror the tangent about an outward unit normal."""
        # A wall leaves the radius alone and preserves `|tangent|`, so both the
        # speed and the angular momentum survive untouched; only the tangent moves.
        tangent = state.tangent - 2 * (state.tangent @ normal) * normal
        return eqx.tree_at(lambda leaf: leaf.tangent, state, tangent)

    @classmethod
    @abc.abstractmethod
    def crossing(cls, normal: Normals, offset: Offsets, state: S) -> tuple[Index, Scalar]:
        """The slot the ray leaves through, and the parameter at which it does."""

    @classmethod
    @abc.abstractmethod
    def advance(cls, state: S, parameter: Scalar) -> tuple[S, Scalar]:
        """Step the ray forward by `parameter`, returning the new state and the 3D length flown.

        The length is exact. Differencing `travel` across the step recovers the
        same number by cancellation and loses its low bits once `travel` grows.
        """


class LinearPath(Path[LinearState]):
    """Straight rays: through a 3D mesh, or through a 2D one read as extruded.

    The crossing parameter is already the 3D distance flown.
    """

    @classmethod
    @override
    def launch(cls, position: Vector, direction: Float[Array, " 3"], travel: Scalar) -> LinearState:
        tangent = direction[: position.shape[-1]]
        return LinearState(position, tangent, travel, jnp.linalg.vector_norm(tangent))

    @classmethod
    @override
    @jax.named_scope("crossing")
    def crossing(cls, normal: Normals, offset: Offsets, state: LinearState) -> tuple[Index, Scalar]:
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

    @classmethod
    @override
    @jax.named_scope("advance")
    def advance(cls, state: LinearState, parameter: Scalar) -> tuple[LinearState, Scalar]:
        return LinearState(
            position=state.position + parameter * state.tangent,
            tangent=state.tangent,
            travel=state.travel + parameter,
            speed=state.speed,
        ), parameter


class AxisymmetricPath(Path[HyperbolicState]):
    """Revolved rays: a 2D mesh read as the `(x, r)` half-plane.

    A straight 3D ray projects to a hyperbola. The crossing parameter is a step
    in the hyperbola's own parameter, not a distance; `advance` converts it.
    """

    @classmethod
    @override
    def simplex_point(cls, key: KeyArray, corners: Float[Array, "ncorner ndim"]) -> Vector:
        # A ring is not uniform in the (x, r) half-plane: the density carries a
        # factor r, making it a mixture over the corners. Drawing one dimension
        # up and folding the spare weight into corner `i` is Dirichlet with
        # alpha_i = 2 and the rest 1, which is that mixture's component.
        key_corner, key_weights = jax.random.split(key)
        radius = corners[:, RADIAL]
        corner = jax.random.choice(key_corner, corners.shape[0], p=radius / radius.sum())
        weights = simplex(key_weights, corners.shape[0])
        return weights[1:].at[corner].add(weights[0]) @ corners

    @classmethod
    @override
    def launch(cls, position: Vector, direction: Float[Array, " 3"], travel: Scalar) -> HyperbolicState:
        # Radius times azimuthal speed. Reading it off the 3D draw keeps full
        # precision for a ray that nearly meets the axis, where recovering it
        # from `1 - |tangent|^2` would cancel away.
        angular_momentum = position[RADIAL] * jnp.abs(direction[2])
        return HyperbolicState(position, direction[:2], travel, angular_momentum)

    @classmethod
    @jax.named_scope("frame")
    def frame(cls, state: HyperbolicState) -> tuple[Vector, Vector, Scalar, Scalar]:
        """The two asymptote directions, and how far along each the ray currently sits.

        The parametrisation is anchored so the ray sits at `v = 1`; a frame
        pinned to the hyperbola's centre instead loses the position to
        cancellation whenever that centre is far away.
        """
        axial_speed, radial_speed = state.tangent[AXIAL], state.tangent[RADIAL]
        # The central-force split: radial motion plus the centrifugal share.
        # A sum, where `1 - axial_speed^2` would cancel for a near-axial ray.
        transverse_squared = radial_speed**2 + (state.angular_momentum / state.position[RADIAL]) ** 2
        transverse_speed = jnp.sqrt(transverse_squared)
        eccentricity = 1 / transverse_speed

        forward = jnp.stack([axial_speed, transverse_speed])
        backward = jnp.stack([-axial_speed, transverse_speed])
        return forward, backward, *cls.asymptote_coordinates(state, eccentricity)

    @classmethod
    @jax.named_scope("asymptote_coordinates")
    def asymptote_coordinates(cls, state: HyperbolicState, eccentricity: Scalar) -> tuple[Scalar, Scalar]:
        """How far along the outgoing and incoming asymptotes the ray sits.

        Both are built as a magnitude-ordered pair first. Only the sum of like
        signs survives a ray that nearly meets the axis, so the larger comes
        from that sum and the smaller from the product the two always make.
        """
        radial_speed, radius = state.tangent[RADIAL], state.position[RADIAL]
        linear_eccentricity = state.angular_momentum * eccentricity**2
        larger = radius * eccentricity * (1 + jnp.abs(radial_speed) * eccentricity)
        smaller = linear_eccentricity**2 / larger

        # A receding ray is further along its outgoing asymptote than its incoming one.
        receding = radial_speed >= 0
        return jnp.where(receding, larger, smaller), jnp.where(receding, smaller, larger)

    @classmethod
    @override
    @jax.named_scope("crossing")
    def crossing(cls, normal: Normals, offset: Offsets, state: HyperbolicState) -> tuple[Index, Scalar]:
        forward, backward, forward_length, backward_length = cls.frame(state)

        # How far the face closes per unit travelled along each asymptote.
        forward_reach = forward_length * (normal @ forward)
        backward_reach = backward_length * (normal @ backward)
        distance = normal @ state.position - offset

        # Solved in `v - 1`, so the ray's own position anchors the quadratic.
        linear = 2 * distance + forward_reach - backward_reach
        discriminant = linear**2 - 8 * distance * forward_reach
        root = jnp.sqrt(jnp.where(discriminant > 0, discriminant, jnp.nan))
        # The root taken away from the cancelling side; both steps divide into it.
        stable = -(linear + jnp.where(linear >= 0, 1.0, -1.0) * root) / 2

        step = jnp.stack([stable / forward_reach, 2 * distance / stable])
        # Outgoing: the sign of dg/dv. A convex trace can leave through the face
        # it entered by, so the derivative and not the ordering picks the root.
        outgoing = (1 + step) ** 2 * forward_reach > backward_reach
        step = jnp.where((step > 0) & outgoing & jnp.isfinite(step), step, jnp.nan)

        best = jnp.nanmin(step, axis=0)
        slot = jnp.nanargmin(best)
        return slot, best[slot]

    @classmethod
    @override
    @jax.named_scope("advance")
    def advance(cls, state: HyperbolicState, parameter: Scalar) -> tuple[HyperbolicState, Scalar]:
        forward, backward, forward_length, backward_length = cls.frame(state)
        # The anchored parameter `v = exp(dtheta)` for `dtheta` the hyperbolic
        # angle swept, so `parameter` is its `expm1`. The forward coordinate
        # stretches by it while the backward one shrinks by the same factor, so
        # their product, and with it the hyperbola, is untouched.
        stretch = 1 + parameter
        grown, shrunk = forward_length * stretch, backward_length / stretch

        moved = parameter / 2 * (forward_length * forward - backward_length * backward / stretch)
        flown = ((grown - shrunk) - (forward_length - backward_length)) / 2
        transverse_speed = forward[RADIAL]
        radial_speed = transverse_speed * (grown - shrunk) / (grown + shrunk)

        return HyperbolicState(
            position=state.position + moved,
            # The axial component is untouched by the march; only a wall moves it.
            tangent=jnp.stack([state.tangent[AXIAL], radial_speed]),
            travel=state.travel + flown,
            angular_momentum=state.angular_momentum,
        ), flown
