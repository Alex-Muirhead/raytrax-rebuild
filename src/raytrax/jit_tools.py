"""Tools for JIT-compiled functions: what causes a recompile, and the batch width and memory of a compile.

The tools know about JAX and the device, not about rays. Nothing here decides
where a ray goes.
"""

import functools
import inspect
from collections.abc import Callable
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
from jax.errors import JaxRuntimeError

# Narrower than this a batch is not worth marching, so a width that has been
# halved down to it is out of memory for some reason other than its width.
MIN_LANES = 1 << 12

# Threads one core keeps resident. Every recent NVIDIA part holds between 1024
# and 2048 per multiprocessor, and a lane count only has to be right to within a
# factor of a few, so the top of that range stands in for all of them.
THREADS_PER_CORE = 2048

# Lanes per resident thread. One apiece already saturates the device; the rest
# is there to hide the latency of the gathers a step makes.
OVERSUBSCRIPTION = 4


def device_width() -> int | None:
    """Give the number of lanes that keeps all cores of the device busy.

    More lanes than this give no more parallelism, but each step pays for all of them.

    Returns
    -------
    lanes
        The lane count, or None if the device does not report its core count.
    """
    cores = getattr(jax.local_devices()[0], "core_count", None)
    if not cores:
        return None
    return max(MIN_LANES, cores * THREADS_PER_CORE * OVERSUBSCRIPTION)


def resident_bytes(lower: Callable[[int], Any], lanes: int) -> int:
    """Give the peak device memory of a compiled march.

    Parameters
    ----------
    lower
        A function that gives the march lowered for a lane count, as
        `eqx.filter_jit(...).lower` returns it.
    lanes
        The lane count to compile the march for.

    Returns
    -------
    bytes
        The peak device memory in bytes.
    """
    # equinox wraps the jax Compiled; the analyses live on the inner one.
    compiled = lower(lanes).compile().compiled
    analysis = compiled.memory_analysis()
    return analysis.temp_size_in_bytes + analysis.argument_size_in_bytes + analysis.output_size_in_bytes


def lane_capacity(
    lower: Callable[[int], Any],
    fraction: float = 0.7,
    probe: tuple[int, int] = (1 << 16, 1 << 18),
) -> int | None:
    """Give the largest lane count of a march that fits in device memory.

    Parameters
    ----------
    lower
        A function that gives the march lowered for a lane count, as
        `eqx.filter_jit(...).lower` returns it. The march must include the refill.
    fraction
        The fraction of the free device memory pool to fill. Increase it on a
        device that has no display and no other process.
    probe
        The two lane counts to compile the march for. Both must be large enough
        that the per-lane memory is more than the fixed memory of the march.

    Returns
    -------
    lanes
        The lane count, or None if the device does not report its memory.
    """
    stats = jax.local_devices()[0].memory_stats()
    if not stats or stats.get("bytes_limit") is None:
        return None
    budget = fraction * (stats["bytes_limit"] - stats["bytes_in_use"])

    small, large = probe
    cheap, dear = resident_bytes(lower, small), resident_bytes(lower, large)
    per_lane = (dear - cheap) / (large - small)
    fixed = cheap - per_lane * small
    return max(MIN_LANES, int((budget - fixed) / per_lane))


def autosized[**P, T](
    argname: str = "lanes",
    width: Callable[[], int | None] = device_width,
    floor: int = MIN_LANES,
) -> Callable[[Callable[P, T]], Callable[P, T]]:
    """Make a decorator that sets a batch width from the device when the caller omits it.

    If the caller gives the width, the callable uses it as given, and an
    out-of-memory error goes to the caller. If the decorator sets the width, it
    halves the width and tries again after an out-of-memory error.

    Parameters
    ----------
    argname
        The name of the parameter that holds the batch width.
    width
        A function that gives the batch width. If it gives None, the callable
        uses its own default.
    floor
        The smallest width to try again with. An out-of-memory error at this
        width goes to the caller.

    Returns
    -------
    decorate
        The decorator.

    Raises
    ------
    TypeError
        If the decorated callable has no parameter named `argname`.
    """

    def decorate(function: Callable[P, T]) -> Callable[P, T]:
        signature = inspect.signature(function)
        if argname not in signature.parameters:
            raise TypeError(f"{function} takes no argument named {argname!r}")

        @functools.wraps(function)
        def sized(*args: P.args, **kwargs: P.kwargs) -> T:
            bound = signature.bind_partial(*args, **kwargs)
            if bound.arguments.get(argname) is not None:
                return function(*args, **kwargs)

            size = width()
            if size is None:
                return function(*args, **kwargs)

            while True:
                bound.arguments[argname] = size
                try:
                    return function(*bound.args, **bound.kwargs)
                except JaxRuntimeError as error:
                    # XLA reports a share of the whole card, so a width it says
                    # fits can still meet a display or another process.
                    if "RESOURCE_EXHAUSTED" not in str(error) or size <= floor:
                        raise
                    size //= 2

        return sized

    return decorate


def jit_dynamic[**P, T](*names: str) -> Callable[[Callable[P, T]], Callable[P, T]]:
    """Make a decorator that compiles a function, with the named arguments always dynamic.

    `eqx.filter_jit` makes a Python scalar argument static, so each new value
    compiles again.

    Parameters
    ----------
    names
        The names of the parameters to pass as arrays.

    Returns
    -------
    decorate
        The decorator.

    Raises
    ------
    TypeError
        If the decorated function has no parameter with one of the names.
    """

    def decorate(function: Callable[P, T]) -> Callable[P, T]:
        signature = inspect.signature(function)
        missing = [name for name in names if name not in signature.parameters]
        if missing:
            raise TypeError(f"{function} takes no arguments named {missing!r}")
        compiled = eqx.filter_jit(function)

        @functools.wraps(function)
        def dynamic(*args: P.args, **kwargs: P.kwargs) -> T:
            bound = signature.bind(*args, **kwargs)
            for name in names:
                if name in bound.arguments:
                    bound.arguments[name] = jnp.asarray(bound.arguments[name])
            return compiled(*bound.args, **bound.kwargs)

        return dynamic

    return decorate
