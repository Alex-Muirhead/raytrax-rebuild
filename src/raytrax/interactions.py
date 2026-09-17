from collections.abc import Mapping
from enum import IntEnum
from types import MappingProxyType
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Float

from raytrax.mesh.boundaries import BoundaryGroups
from raytrax.paths import Path, RayState
from raytrax.shapes import Flag, Index, KeyArray, Scalar, Vector


class Kind(IntEnum):
    """The action of a ray when it hits a face of a boundary group.

    Attributes
    ----------
    ESCAPE
        The ray leaves the domain and takes its remaining energy with it.
    ABSORB
        The ray stops and deposits its remaining energy on the face.
    SPECULAR
        The ray reflects as in a mirror about the face normal.
    DIFFUSE
        The ray goes back into the domain in a new Lambertian direction.
    """

    ESCAPE = 0
    ABSORB = 1
    SPECULAR = 2
    DIFFUSE = 3


class Interactions(eqx.Module):
    """The boundary parameters of each group, by group index.

    Attributes
    ----------
    kind
        The interaction of each group. This field is static, so a change to it
        causes a recompile.
    reflectivity
        The fraction of the energy that each group reflects. Only `SPECULAR`
        and `DIFFUSE` groups use it; the face absorbs the remainder.
    """

    kind: tuple[Kind, ...] = eqx.field(static=True)
    reflectivity: Float[Array, " ngroup"]


def build(
    groups: BoundaryGroups,
    kinds: Mapping[str, Kind],
    reflectivity: Mapping[str, float] = MappingProxyType({}),
) -> Interactions:
    """Make the interaction table for a set of boundary groups.

    Parameters
    ----------
    groups
        The boundary groups of the mesh.
    kinds
        The interaction of each named group. A group that is not in this map escapes.
    reflectivity
        The reflectivity of each named group. A group that is not in this map has
        a reflectivity of 1.

    Returns
    -------
    interactions
        The interaction table, by group index.

    Raises
    ------
    ValueError
        If a name is not a group of the mesh.
    """
    kind = np.full(len(groups.names), Kind.ESCAPE, dtype=np.int64)
    rho = np.ones(len(groups.names))

    for name, value in kinds.items():
        kind[groups.index(name)] = value
    for name, value in reflectivity.items():
        rho[groups.index(name)] = value

    return Interactions(kind=tuple(Kind(value) for value in kind), reflectivity=jnp.asarray(rho))


@jax.named_scope("interaction")
def apply(
    path: Path[Any],
    interactions: Interactions,
    group: Index,
    key: KeyArray,
    state: RayState,
    normal: Vector,
) -> tuple[RayState, Scalar, Flag]:
    """Apply the interaction of a group to a ray that hits a boundary face.

    Parameters
    ----------
    path
        The ray kinematics of the run.
    interactions
        The interaction table.
    group
        The group index of the face.
    key
        The PRNG key for a diffuse direction.
    state
        The ray at the face.
    normal
        The unit normal of the face, out of the cell that the ray leaves.

    Returns
    -------
    state
        The ray after the interaction.
    absorbed
        The fraction of the ray energy that the face absorbs.
    terminated
        True if the ray stops at the face.
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

    # Under `vmap` a `switch` evaluates every branch it is given, so keeping a
    # branch no group asks for costs every ray a Lambertian draw per step.
    handler = {Kind.ESCAPE: escape, Kind.ABSORB: absorb, Kind.SPECULAR: specular, Kind.DIFFUSE: diffuse}
    present = sorted(set(interactions.kind))
    slot = jnp.asarray([present.index(kind) for kind in interactions.kind])
    return jax.lax.switch(slot[group], [handler[kind] for kind in present], state)
