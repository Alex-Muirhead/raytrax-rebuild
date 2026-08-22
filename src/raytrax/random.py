import jax
import jax.numpy as jnp


def sphere(key, ndim: int, shape: tuple[int, ...] = ()):
    """Sample a direction uniformly on the unit sphere in N-dimensional space."""
    samples = jax.random.normal(key, shape=(*shape, ndim))
    return samples / jnp.linalg.vector_norm(samples, axis=-1, keepdims=True)


def simplex(key, ndim: int, shape: tuple[int, ...] = ()):
    """Sample barycentric coordinates uniformly over an N-simplex."""
    # Sorted uniforms cut [0, 1] into ndim+1 spans that are already Dirichlet(1).
    samples = jnp.sort(jax.random.uniform(key, shape=(*shape, ndim)), axis=-1)
    return jnp.diff(samples, axis=-1, prepend=0.0, append=1.0)


def cosine_hemisphere(key, normal):
    """Sample a cosine-weighted (Lambertian) direction about `normal`.

    Exact in 3D, which is where every caller uses it: `Path.lambertian` embeds an
    in-plane normal before drawing, rather than sampling in the plane.
    """
    direction = normal + sphere(key, ndim=normal.shape[-1])
    return direction / jnp.linalg.vector_norm(direction, axis=-1, keepdims=True)
