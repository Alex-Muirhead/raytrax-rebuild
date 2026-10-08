"""Time `transport.march` over a mesh, lane and path ladder.

    uv run python -m raytrax_examples.benchmark --case slab_planar --lanes 65536
    uv run python -m raytrax_examples.benchmark --ladder --json bench.json

Reports compile time and run time separately, because `march` carries no
`static_argnums` and takes its statics implicitly, so every distinct
(case, lanes, steps) is a fresh compile.

Timing follows the JAX guide: warm up once, then block on the whole returned
pytree, not one array, or the numbers measure async dispatch instead of compute.

Set `JAX_PLATFORMS=cpu` to run the same ladder on the host. The performance
example runs the same `measure` over every case and writes it for the thesis.
"""

import argparse
import json
import os
import platform
import statistics
import subprocess
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import jax

from raytrax import transport
from raytrax.problem import Problem
from raytrax_examples.cases import Case, cylinder, plane, reentry, slab

# The cases a command line can name, each given the absorption coefficient.
CASES: dict[str, Callable[[float], Case]] = {
    "slab_planar": lambda kappa: slab.case(kappa=kappa),
    "slab_axisymmetric": lambda kappa: slab.case(axisymmetric=True, kappa=kappa),
    "plane": lambda kappa: plane.case(kappa=kappa),
    "plane_axisymmetric": lambda kappa: plane.case(axisymmetric=True, kappa=kappa),
    "cylinder_coarse": lambda kappa: cylinder.case(kappa=kappa, mesh_size=0.6),
    "cylinder": lambda kappa: cylinder.case(kappa=kappa),
    "reentry": lambda kappa: reentry.case(kappa=kappa),
}

# A step limit that no trace here reaches; the trace stops when its rays do.
MAX_TRACE_STEPS = 1 << 30


def make_rays(case: Problem, lanes: int, seed: int) -> transport.Rays:
    """Launch `lanes` rays from the source, outside the timed region."""
    return transport.launch(jax.random.key(seed), case, lanes)


def peak_mib() -> float | None:
    """Peak device bytes so far, in MiB, or None on a backend that does not report it."""
    stats = jax.local_devices()[0].memory_stats()
    if not stats or stats.get("peak_bytes_in_use") is None:
        return None
    return stats["peak_bytes_in_use"] / 2**20


def measure(case: Problem, lanes: int, steps: int, repeats: int, seed: int, extinction: float) -> dict[str, Any]:
    """Compile once, then time `repeats` marches, blocking on the whole result."""
    rays = make_rays(case, lanes, seed)
    key = jax.random.key(seed + 1)

    def march():
        return transport.march(key, rays, case, steps=steps, floor=extinction)

    start = time.perf_counter()
    final, deposit = jax.block_until_ready(march())
    first = time.perf_counter() - start

    runs = []
    for _ in range(repeats):
        start = time.perf_counter()
        jax.block_until_ready(march())
        runs.append(time.perf_counter() - start)

    median = statistics.median(runs)
    return {
        "lanes": lanes,
        "steps": steps,
        "extinction": extinction,
        "cells": int(case.scene.num_cells),
        "faces": int(case.scene.num_faces),
        "slots": int(case.scene.cells.normal.shape[1]),
        # The first call is compile plus one run; the difference is the compile.
        "compile_s": first - median,
        "median_s": median,
        "min_s": min(runs),
        "spread_s": max(runs) - min(runs),
        "ms_per_step": median / steps * 1e3,
        "lane_steps_per_s": lanes * steps / median,
        "peak_mib": peak_mib(),
        "alive_at_end": int(final.alive.sum()),
        "deposited_fraction": float(deposit.cells.sum() + deposit.faces.sum()) / lanes,
    }


def measure_trace(case: Problem, num_rays: int, lanes: int, repeats: int, seed: int) -> dict[str, Any]:
    """Compile from cold, then time `repeats` whole traces, blocking on the whole tally.

    The run goes to the default device, so call it inside `jax.default_device`
    with `case` already put on that device.

    Raises
    ------
    RuntimeError
        If the step limit stops the trace before it launches every ray.
    """

    def trace(offset):
        # The default step limit stops a narrow batch of many rays early.
        return transport.launch_and_march(
            jax.random.key(seed + offset), case, num_rays, lanes=lanes, max_steps=MAX_TRACE_STEPS
        )

    # Without this a lane count seen before reuses its compile, and the first call times only the run.
    jax.clear_caches()
    start = time.perf_counter()
    tally = jax.block_until_ready(trace(0))
    first = time.perf_counter() - start

    runs = []
    for offset in range(1, repeats + 1):
        start = time.perf_counter()
        jax.block_until_ready(trace(offset))
        runs.append(time.perf_counter() - start)

    if int(tally.launched) != num_rays or bool(tally.rays.alive.any()):
        raise RuntimeError(f"the step limit stopped the trace after {int(tally.launched)} of {num_rays} rays")

    median = statistics.median(runs)
    deposit = tally.estimate(case)
    budget = deposit.cells.sum() + deposit.faces.sum() + deposit.escaped + tally.in_flight(case)
    return {
        "rays": num_rays,
        "lanes": lanes,
        # The first call is compile plus one run; the difference is the compile.
        "compile_s": first - median,
        "median_s": median,
        "min_s": min(runs),
        "max_s": max(runs),
        "rays_per_s": num_rays / median,
        "energy_error": float(budget / case.sources.total - 1),
    }


def environment(device: jax.Device | None = None) -> dict[str, Any]:
    """Enough about this machine to compare a run against another one, for `device` or the first device."""
    device = device or jax.local_devices()[0]
    driver = ""
    try:
        if device.platform == "gpu":
            driver = subprocess.run(
                ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
                capture_output=True,
                text=True,
                timeout=10,
                check=False,
            ).stdout.strip()
    except OSError, subprocess.SubprocessError:
        pass
    return {
        "device_kind": device.device_kind,
        "platform": device.platform,
        "driver": driver,
        "host": platform.node(),
        "cpu_count": os.cpu_count(),
        "python": platform.python_version(),
        "jax": jax.__version__,
        "x64": bool(jax.config.jax_enable_x64),
    }


HEADER = f"{'lanes':>9}{'compile_s':>11}{'median_s':>10}{'min_s':>10}{'ms/step':>10}{'Mlanestep/s':>13}{'peakMiB':>9}{'alive':>8}"


def row(record: dict[str, Any]) -> str:
    peak = "-" if record["peak_mib"] is None else f"{record['peak_mib']:.1f}"
    return (
        f"{record['lanes']:>9}{record['compile_s']:>11.3f}{record['median_s']:>10.4f}"
        f"{record['min_s']:>10.4f}{record['ms_per_step']:>10.4f}"
        f"{record['lane_steps_per_s'] / 1e6:>13.1f}{peak:>9}{record['alive_at_end']:>8}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--case", default="slab_planar", choices=sorted(CASES))
    parser.add_argument("--lanes", type=int, default=65_536)
    parser.add_argument("--ladder", action="store_true", help="sweep lanes in powers of two instead")
    parser.add_argument("--from-power", type=int, default=13)
    parser.add_argument("--to-power", type=int, default=22)
    parser.add_argument("--steps", type=int, default=400)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--extinction", type=float, default=1e-8, help="share of a quantum below which a ray stops")
    parser.add_argument("--kappa", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--x32", action="store_true", help="undo the package's x64 default, to price fp64")
    parser.add_argument("--json", type=Path, default=None)
    args = parser.parse_args()

    if args.x32:
        # `raytrax/__init__` turns x64 on at import; the flag is read when an
        # array is made, so overriding it here is enough.
        jax.config.update("jax_enable_x64", False)

    env = environment()
    print(f"# {env['device_kind']} ({env['platform']}) | jax {env['jax']} | x64={env['x64']}")

    case = CASES[args.case](args.kappa).problem
    lanes = [1 << p for p in range(args.from_power, args.to_power + 1)] if args.ladder else [args.lanes]
    print(f"# {args.case}: cells={case.scene.num_cells} faces={case.scene.num_faces} steps={args.steps}")
    print(HEADER)

    records = []
    for count in lanes:
        try:
            record = measure(case, count, args.steps, args.repeats, args.seed, args.extinction)
        except Exception as error:  # noqa: BLE001 - an OOM ends the ladder, not the run
            print(f"{count:>9}  FAILED  {type(error).__name__}: {str(error).splitlines()[0][:90]}")
            break
        records.append(record)
        print(row(record))

    if args.json:
        args.json.write_text(json.dumps({"environment": env, "case": args.case, "runs": records}, indent=2) + "\n")
        print(f"# wrote {args.json}")
