import jax
import jax.numpy as jnp
from jaxtyping import Array, Float

from raytrax.shapes import KeyArray


def sphere(key: KeyArray, ndim: int, shape: tuple[int, ...] = ()) -> Float[Array, "*shape ndim"]:
    """Sample a direction uniformly on the unit sphere.

    Parameters
    ----------
    key
        The PRNG key for the sample.
    ndim
        The number of dimensions of the space.
    shape
        The batch shape of the samples.

    Returns
    -------
    direction
        A unit vector for each batch entry.
    """
    samples = jax.random.normal(key, shape=(*shape, ndim))
    return samples / jnp.linalg.vector_norm(samples, axis=-1, keepdims=True)


def fixed_sorted(samples: Float[Array, "... n"]) -> Float[Array, "... n"]:
    """Sort the last axis in ascending order.

    The result is identical to `jnp.sort`. Use this function under `vmap`, where
    it is much faster for a length of 3 or less.

    Parameters
    ----------
    samples
        The values to sort.

    Returns
    -------
    sorted
        The values, sorted along the last axis.
    """

    def swap(x: Array, y: Array) -> tuple[Array, Array]:
        return jnp.minimum(x, y), jnp.maximum(x, y)

    match samples.shape[-1]:
        case 1:
            return samples
        case 2:
            a, b = jnp.unstack(samples, axis=-1)
            a, b = swap(a, b)
            return jnp.stack([a, b], axis=-1)
        case 3:
            a, b, c = jnp.unstack(samples, axis=-1)
            a, b = swap(a, b)
            b, c = swap(b, c)
            a, b = swap(a, b)
            return jnp.stack([a, b, c], axis=-1)
        case _:
            return jnp.sort(samples, axis=-1)


def simplex(key: KeyArray, ndim: int, shape: tuple[int, ...] = ()) -> Float[Array, "*shape ncorner"]:
    """Sample barycentric coordinates uniformly over a simplex.

    Parameters
    ----------
    key
        The PRNG key for the sample.
    ndim
        The dimension of the simplex, which has `ndim + 1` corners.
    shape
        The batch shape of the samples.

    Returns
    -------
    weights
        A non-negative weight for each corner. The weights sum to exactly 1.
    """
    # Sorted uniforms cut [0, 1] into ndim+1 spans that are already Dirichlet(1).
    # The gaps telescope, so they sum to exactly one and the sampled point is an
    # exact convex combination; see docs/simplex.typ.
    samples = fixed_sorted(jax.random.uniform(key, shape=(*shape, ndim)))
    return jnp.diff(samples, axis=-1, prepend=0.0, append=1.0)


def cosine_hemisphere(key: KeyArray, normal: Float[Array, "... ndim"]) -> Float[Array, "... ndim"]:
    """Sample a cosine-weighted (Lambertian) direction about a normal.

    The distribution is correct only in 3D.

    Parameters
    ----------
    key
        The PRNG key for the sample.
    normal
        The unit normal at the center of the hemisphere.

    Returns
    -------
    direction
        A unit vector in the hemisphere of `normal`.
    """
    direction = normal + sphere(key, ndim=normal.shape[-1])
    return direction / jnp.linalg.vector_norm(direction, axis=-1, keepdims=True)
