"""How wide a batch to march, for the device marching it.

Nothing here decides where a ray goes, only how many go at once.
"""

import functools
import inspect
from collections.abc import Callable
from typing import Any

import jax
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
    """Lanes it takes to keep this device's cores busy, or None if it does not report them.

    A step costs the same whether a lane is alive or dead, so lanes past this
    buy no parallelism and are paid for in full at every step of the march.
    """
    cores = getattr(jax.local_devices()[0], "core_count", None)
    if not cores:
        return None
    return max(MIN_LANES, cores * THREADS_PER_CORE * OVERSUBSCRIPTION)


def resident_bytes(march: Callable[..., Any], lanes: int, **traced: Any) -> int:
    """Device bytes `march` holds at its peak when run over `lanes` lanes.

    `march` is the jitted march to price, and `traced` every argument it takes
    except the lane count.
    """
    # pyrefly: ignore[missing-attribute] - it does not model what filter_jit returns
    lowered = march.lower(lanes=lanes, **traced)
    # equinox wraps the jax Compiled; the analyses live on the inner one.
    compiled = lowered.compile().compiled
    analysis = compiled.memory_analysis()
    return analysis.temp_size_in_bytes + analysis.argument_size_in_bytes + analysis.output_size_in_bytes


def lane_capacity(
    march: Callable[..., Any],
    fraction: float = 0.7,
    probe: tuple[int, int] = (1 << 16, 1 << 18),
    **traced: Any,
) -> int | None:
    """The most lanes `march` fits in device memory, or None if the device does not report it.

    `march` is the jitted march to price. `fraction` is the share of the pool to
    fill: the pool XLA reports is a share of the whole card rather than of what
    is free on it, so the default leaves room for a display or a second process,
    and a dedicated device can raise it. `probe` is the pair of lane counts
    compiled to price a lane, and wants to sit above the few MiB a march costs
    whatever the lane count. `traced` is every argument `march` takes except the
    lane count, and wants a ray count past the probe widths, so that the refill
    an automatic lane count always ends up using is compiled in.
    """
    stats = jax.local_devices()[0].memory_stats()
    if not stats or stats.get("bytes_limit") is None:
        return None
    budget = fraction * (stats["bytes_limit"] - stats["bytes_in_use"])

    small, large = probe
    cheap, dear = resident_bytes(march, small, **traced), resident_bytes(march, large, **traced)
    per_lane = (dear - cheap) / (large - small)
    fixed = cheap - per_lane * small
    return max(MIN_LANES, int((budget - fixed) / per_lane))


def autosized[**P, T](
    argname: str = "lanes",
    width: Callable[[], int | None] = device_width,
    floor: int = MIN_LANES,
) -> Callable[[Callable[P, T]], Callable[P, T]]:
    """Decorate a callable so that leaving `argname` out sizes it to the device.

    `argname` is the parameter holding the batch width, by name, so the callable
    is free to call it whatever it likes. `width` supplies the size when the
    caller leaves that parameter out or passes None, and a `width` that returns
    None leaves it out too, for the callable's own default to answer. `floor` is
    the narrowest width worth retrying at.

    A width the caller passes is used exactly as given, and an overrun on it
    raises. Only a width this chose is halved and tried again.
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
