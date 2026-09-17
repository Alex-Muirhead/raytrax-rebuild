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
    """The position and direction of a ray.

    Attributes
    ----------
    position
        The current position of the ray in the mesh.
    tangent
        The in-plane components of the unit 3D direction. In 2D, its magnitude
        is less than 1 by the out-of-plane component.
    travel
        The 3D distance that the ray has flown since launch.
    """

    position: Float[Array, "... ndim"]
    tangent: Float[Array, "... ndim"]
    travel: Float[Array, "..."]


class LinearState(RayState):
    """The state of a straight ray.

    Attributes
    ----------
    speed
        The magnitude of `tangent`, which is constant for the life of the ray.
    """

    speed: Float[Array, "..."]


class HyperbolicState(RayState):
    """The state of a ray in the (x, r) half-plane of an axisymmetric mesh.

    Attributes
    ----------
    angular_momentum
        The radius times the magnitude of the azimuthal component of the unit
        direction. It is constant along the ray and through a reflection.
    """

    angular_momentum: Float[Array, "..."]


class Path[S: RayState](eqx.Module):
    """The ray kinematics: how a ray crosses a cell, and the distance it flies.

    All methods are classmethods. A subclass holds no state.
    """

    # Classmethods: a mode holds no state, and anything that varies between
    # modes is class data. Dispatch is still on the instance's type.
    @classmethod
    def isotropic(cls, key: KeyArray) -> Float[Array, " 3"]:
        """Sample a 3D direction uniformly on the unit sphere.

        Parameters
        ----------
        key
            The PRNG key for the sample.

        Returns
        -------
        direction
            A 3D unit vector.
        """
        return sphere(key, SPATIAL_DIM)

    @classmethod
    def lambertian(cls, key: KeyArray, normal: Vector) -> Float[Array, " 3"]:
        """Sample a 3D cosine-weighted direction about an in-plane normal.

        Parameters
        ----------
        key
            The PRNG key for the sample.
        normal
            The in-plane unit normal at the center of the hemisphere.

        Returns
        -------
        direction
            A 3D unit vector in the hemisphere of `normal`.
        """
        return cosine_hemisphere(key, jnp.pad(normal, (0, SPATIAL_DIM - normal.shape[-1])))

    @classmethod
    def simplex_point(cls, key: KeyArray, corners: Float[Array, "ncorner ndim"]) -> Vector:
        """Sample a point uniformly in the simplex that the corners make.

        The density is uniform in the volume that the mesh represents, which is
        not always the plane of the mesh.

        Parameters
        ----------
        key
            The PRNG key for the sample.
        corners
            The corner positions of the simplex.

        Returns
        -------
        point
            A position in the simplex.
        """
        return simplex(key, corners.shape[0] - 1) @ corners

    @classmethod
    @abc.abstractmethod
    def launch(cls, position: Vector, direction: Float[Array, " 3"], travel: Scalar) -> S:
        """Make a ray state from a position and a 3D direction.

        Parameters
        ----------
        position
            The start position of the ray in the mesh.
        direction
            The 3D unit direction of the ray.
        travel
            The distance that the ray has already flown.

        Returns
        -------
        state
            The ray state.
        """

    @classmethod
    def reflect(cls, state: S, normal: Vector) -> S:
        """Reflect a ray as in a mirror.

        Parameters
        ----------
        state
            The ray at the mirror.
        normal
            The in-plane unit normal of the mirror.

        Returns
        -------
        state
            The ray after the reflection.
        """
        # A wall leaves the radius alone and preserves `|tangent|`, so both the
        # speed and the angular momentum survive untouched; only the tangent moves.
        tangent = state.tangent - 2 * (state.tangent @ normal) * normal
        return eqx.tree_at(lambda leaf: leaf.tangent, state, tangent)

    @classmethod
    @abc.abstractmethod
    def crossing(cls, normal: Normals, offset: Offsets, state: S) -> tuple[Index, Scalar]:
        """Find the face slot through which a ray leaves a cell.

        Parameters
        ----------
        normal
            The outward unit normal of each face slot of the cell.
        offset
            The plane constant of each face slot of the cell.
        state
            The ray in the cell.

        Returns
        -------
        slot
            The face slot through which the ray leaves.
        parameter
            The path parameter at the exit, for `advance`. It is not always a distance.
        """

    @classmethod
    @abc.abstractmethod
    def advance(cls, state: S, parameter: Scalar) -> tuple[S, Scalar]:
        """Move a ray forward along its path.

        Parameters
        ----------
        state
            The ray before the move.
        parameter
            The path parameter of the move, as `crossing` gives it.

        Returns
        -------
        state
            The ray after the move.
        flown
            The 3D distance of the move. Use this value, not the difference in
            `travel`, which is less accurate.
        """


class LinearPath(Path[LinearState]):
    """The kinematics of straight rays, in a 3D mesh or an extruded 2D mesh.

    The crossing parameter is the 3D distance.
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
    """The kinematics of rays in an axisymmetric mesh, as the (x, r) half-plane.

    A straight 3D ray is a hyperbola in the half-plane. The crossing parameter
    is a step in the parameter of the hyperbola, not a distance. Use it with a
    revolved geometry.
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
        """Give the asymptote directions of the hyperbola of a ray, and the ray coordinates on them.

        The ray is at the parameter value `v = 1`.

        Parameters
        ----------
        state
            The ray.

        Returns
        -------
        forward
            The direction of the outgoing asymptote.
        backward
            The direction of the incoming asymptote.
        forward_length
            The coordinate of the ray along the outgoing asymptote.
        backward_length
            The coordinate of the ray along the incoming asymptote.
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
        """Give the ray coordinates on the outgoing and incoming asymptotes of its hyperbola.

        Parameters
        ----------
        state
            The ray.
        eccentricity
            The eccentricity of the hyperbola.

        Returns
        -------
        forward_length
            The coordinate of the ray along the outgoing asymptote.
        backward_length
            The coordinate of the ray along the incoming asymptote.
        """
        # Built as a magnitude-ordered pair first. Only the sum of like signs
        # survives a ray that nearly meets the axis, so the larger comes from
        # that sum and the smaller from the product the two always make.
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
