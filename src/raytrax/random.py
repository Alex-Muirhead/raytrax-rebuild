import jax
import jax.numpy as jnp
from jaxtyping import Array, Float

from raytrax.shapes import KeyArray


def sphere(key: KeyArray, ndim: int, shape: tuple[int, ...] = ()) -> Float[Array, "*shape ndim"]:
    """Sample a direction uniformly on the unit sphere in N-dimensional space."""
    samples = jax.random.normal(key, shape=(*shape, ndim))
    return samples / jnp.linalg.vector_norm(samples, axis=-1, keepdims=True)


def fixed_sorted(samples: Float[Array, "... n"]) -> Float[Array, "... n"]:
    """Sort the last axis, giving exactly what `jnp.sort` would give.

    A comparator network up to length 3, which is every length the meshes ask
    for. `jnp.sort` under `vmap` becomes a segmented radix sort over the whole
    batch, which for three elements costs about 2000 times as much.
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
    """Sample barycentric coordinates uniformly over an N-simplex."""
    # Sorted uniforms cut [0, 1] into ndim+1 spans that are already Dirichlet(1).
    # The gaps telescope, so they sum to exactly one and the sampled point is an
    # exact convex combination; see docs/simplex.typ.
    samples = fixed_sorted(jax.random.uniform(key, shape=(*shape, ndim)))
    return jnp.diff(samples, axis=-1, prepend=0.0, append=1.0)


def cosine_hemisphere(key: KeyArray, normal: Float[Array, "... ndim"]) -> Float[Array, "... ndim"]:
    """Sample a cosine-weighted (Lambertian) direction about `normal`.

    Only exact in 3D, which is where every caller uses it.
    """
    direction = normal + sphere(key, ndim=normal.shape[-1])
    return direction / jnp.linalg.vector_norm(direction, axis=-1, keepdims=True)
