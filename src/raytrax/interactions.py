from collections.abc import Mapping
from enum import IntEnum
from types import MappingProxyType
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Float, Int

from raytrax.mesh.boundaries import BoundaryGroups
from raytrax.paths import Path, RayState
from raytrax.shapes import Flag, Index, KeyArray, Scalar, Vector


class Kind(IntEnum):
    """What a ray does when it reaches a boundary group."""

    ESCAPE = 0  # Leaves the domain, carrying its remaining energy away
    ABSORB = 1  # Terminates, depositing its remaining energy on the exit face
    SPECULAR = 2  # Mirrors about the face normal
    DIFFUSE = 3  # Resamples a Lambertian direction back into the domain


class Interactions(eqx.Module):
    """Per-boundary-group parameters, indexed by group id."""

    kind: Int[Array, " ngroup"]
    reflectivity: Float[Array, " ngroup"]


def build(
    groups: BoundaryGroups,
    kinds: Mapping[str, Kind],
    reflectivity: Mapping[str, float] = MappingProxyType({}),
) -> Interactions:
    """Assemble the interaction table. Groups left out escape."""
    kind = np.full(len(groups.names), Kind.ESCAPE, dtype=np.int64)
    rho = np.ones(len(groups.names))

    for name, value in kinds.items():
        kind[groups.index(name)] = value
    for name, value in reflectivity.items():
        rho[groups.index(name)] = value

    return Interactions(kind=jnp.asarray(kind), reflectivity=jnp.asarray(rho))


def apply(
    path: Path[Any],
    interactions: Interactions,
    group: Index,
    key: KeyArray,
    state: RayState,
    normal: Vector,
) -> tuple[RayState, Scalar, Flag]:
    """Resolve a boundary hit into (state, absorbed fraction, terminated).

    `normal` is the outward unit normal of the face the ray left through.
    """
    rho = interactions.reflectivity[group]

    def escape(state: RayState) -> tuple[RayState, Scalar, Flag]:
        return state, jnp.zeros(()), jnp.bool_(True)

    def absorb(state: RayState) -> tuple[RayState, Scalar, Flag]:
        return state, jnp.ones(()), jnp.bool_(True)

    def specular(state: RayState) -> tuple[RayState, Scalar, Flag]:
        return path.reflect(state, normal), 1 - rho, jnp.bool_(False)

    def diffuse(state: RayState) -> tuple[RayState, Scalar, Flag]:
        tangent = path.lambertian(key, -normal)
        return path.launch(state.position, tangent, state.travel), 1 - rho, jnp.bool_(False)

    return jax.lax.switch(interactions.kind[group], [escape, absorb, specular, diffuse], state)
