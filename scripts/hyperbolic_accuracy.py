"""Regenerate the accuracy table in docs/hyperbolic.typ.

    uv run python scripts/hyperbolic_accuracy.py

Measures AxisymmetricPath against a longdouble reference over random rays in
random bounded convex cells, as a function of the ray's closest approach to the
axis. That approach is what controls the conditioning: it is the quantity the
asymptote frame carries squared, so a naive frame loses it below sqrt(eps).

The reference bisects max_i g_i(t) on the projected trace, which needs no
hyperbola at all -- r(t) is the square root of a quadratic -- so it is
independent of how the kernel chooses to parametrise the arc.
"""

import argparse

import jax
import jax.numpy as jnp
import numpy as np

from raytrax.paths import AxisymmetricPath

LD = np.longdouble
PATH = AxisymmetricPath()


def meridional(t, ray, dtype=LD):
    """The projected trace (x, r) of a straight 3D ray."""
    x0, r0, axial, radial = (dtype(v) for v in ray)
    t = np.asarray(t, dtype=dtype)
    return x0 + axial * t, np.sqrt((dtype(1) - axial**2) * t * t + 2 * r0 * radial * t + r0**2)


def gap(t, normal, offset, ray):
    """max_i (n_i . p(t) - c_i): negative inside the cell."""
    x, r = meridional(t, ray)
    n, c = np.asarray(normal, LD), np.asarray(offset, LD)
    return (n[:, 0] * np.asarray(x)[..., None] + n[:, 1] * np.asarray(r)[..., None] - c).max(axis=-1)


def scanned_exit(normal, offset, ray, horizon=12.0, samples=3000, refine=80):
    grid = np.linspace(LD(0), LD(horizon), samples)[1:]
    crossed = np.flatnonzero(gap(grid, normal, offset, ray) > 0)
    if crossed.size == 0:
        return None
    hi, lo = grid[crossed[0]], (grid[crossed[0] - 1] if crossed[0] else LD(0))
    for _ in range(refine):
        mid = (lo + hi) / 2
        lo, hi = (lo, mid) if gap(mid, normal, offset, ray) > 0 else (mid, hi)
    return float((lo + hi) / 2)


def bounded_cell(rng, inside, num_faces=4):
    """Normals that positively span, so the cell is closed and a ray must exit."""
    while True:
        angles = np.sort(rng.uniform(0, 2 * np.pi, num_faces))
        if np.diff(np.concatenate([angles, angles[:1] + 2 * np.pi])).max() < np.pi - 0.15:
            break
    normal = np.stack([np.cos(angles), np.sin(angles)], axis=-1)
    return normal, normal @ inside + rng.uniform(0.3, 1.5, num_faces)


def direction_approaching(rng, ratio):
    """A direction whose closest approach to the axis is `ratio` times the radius."""
    axial = rng.uniform(-0.9, 0.9)
    transverse = np.sqrt(1 - axial**2)
    phi = np.arcsin(np.clip(ratio, 0.0, 1.0)) * rng.choice([-1, 1])
    return np.array([axial, rng.choice([-1, 1]) * transverse * np.cos(phi), transverse * np.sin(phi)])


@jax.jit
def exit_position(position, direction, normal, offset):
    state = PATH.launch(position, direction, jnp.zeros(()))
    _, step = PATH.crossing(normal, offset, state)
    return PATH.advance(state, step).position


def measure(ratio, trials, seed):
    rng = np.random.default_rng(seed)
    errors, failures, used = [], 0, 0
    for _ in range(trials):
        r0, x0 = rng.uniform(0.2, 2.0), rng.uniform(-1.0, 1.0)
        direction = rng.normal(size=3) if ratio is None else direction_approaching(rng, ratio)
        direction = direction / np.linalg.norm(direction)
        ray = (x0, r0, direction[0], direction[1])
        normal, offset = bounded_cell(rng, np.array([x0, r0]))
        want = scanned_exit(normal, offset, ray)
        if want is None or want < 1e-9:
            continue
        used += 1
        expected = np.array([float(c) for c in meridional(LD(want), ray)])
        got = np.asarray(
            exit_position(jnp.array([x0, r0]), jnp.asarray(direction), jnp.asarray(normal), jnp.asarray(offset))
        )
        if not np.all(np.isfinite(got)):
            failures += 1
        else:
            errors.append(float(np.linalg.norm(got - expected)))
            failures += errors[-1] > 1e-9
    return np.array(errors), failures, used


def frequency(samples, seed):
    """P(approach < x) for an isotropic draw, which is (2/pi) arcsin(x)."""
    rng = np.random.default_rng(seed)
    d = rng.normal(size=(samples, 3))
    d /= np.linalg.norm(d, axis=1, keepdims=True)
    return np.abs(d[:, 2]) / np.sqrt(d[:, 1] ** 2 + d[:, 2] ** 2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trials", type=int, default=500)
    parser.add_argument("--samples", type=int, default=2_000_000, help="draws for the frequency check")
    parser.add_argument("--seed", type=int, default=909)
    args = parser.parse_args()

    approach = frequency(args.samples, args.seed)
    print(f"closest approach a/r over {args.samples:,} isotropic draws")
    print(f"  mean   {approach.mean():.6f}  expected 2/pi = {2 / np.pi:.6f}")
    print(f"  median {np.median(approach):.6f}  expected sin(pi/4) = {np.sin(np.pi / 4):.6f}\n")

    header = f"{'a/r':>10}{'P(below)':>12}{'median':>12}{'max':>12}{'failures':>12}"
    print(header)
    print("-" * len(header))
    for exponent in (None, 2, 4, 8, 12, 16):
        ratio = None if exponent is None else 10.0**-exponent
        errors, failures, used = measure(ratio, args.trials, args.seed + (exponent or 0))
        label = "isotropic" if ratio is None else f"1e-{exponent}"
        chance = "-" if ratio is None else f"{2 * ratio / np.pi:.1e}"
        print(f"{label:>10}{chance:>12}{np.median(errors):>12.1e}{errors.max():>12.1e}{f'{failures} / {used}':>12}")
