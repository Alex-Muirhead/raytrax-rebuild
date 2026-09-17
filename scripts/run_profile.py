"""Profile one whole run, stage by stage, and write a trace for xprof.

    uv run --group profiling python scripts/run_profile.py --case reentry
    uv run --group profiling xprof --logdir profile/     # or tensorboard

Splits a run the way the pipeline is written -- mesh read, scene build, compile,
march -- so a cost lands on a stage instead of on `trace` as a whole, and prints
the device kernels the march spent its time in. `scripts/kernel_profile.py` is
the same kernel table for `march` alone.

The trace it writes covers one warmed `trace` call and opens in xprof or
tensorboard. Needs the `profiling` dependency group. Meshes come from
`examples/meshes/generate.py` and are not committed.
"""

import argparse
import contextlib
import json
import statistics
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import jax
from benchmark import CASES, environment, prepare
from kernel_profile import kernel_rows

from raytrax import interactions, sources, transport, tuning
from raytrax import scene as scene_module
from raytrax.io import vtk
from raytrax.mesh import geometry
from raytrax.problem import Problem

# The host-side pipeline, as `prepare` calls it. Patched rather than inlined so
# there is one description of the pipeline and the timings cannot drift from it.
BUILD_STAGES = (
    (vtk, "read"),
    (geometry, "revolve"),
    (scene_module, "build"),
    (interactions, "build"),
    (sources, "build"),
)


@contextlib.contextmanager
def timed(stages: tuple[tuple[Any, str], ...]) -> Iterator[dict[str, float]]:
    """Time each named module function for the duration of the block."""
    elapsed: dict[str, float] = {}
    original = [(module, name, getattr(module, name)) for module, name in stages]

    def wrap(module: Any, name: str, function: Any) -> Any:
        def timing(*args: Any, **kwargs: Any) -> Any:
            start = time.perf_counter()
            result = function(*args, **kwargs)
            # Dispatch is asynchronous, so a stage that lands on the device is
            # only timed once the device has caught up.
            jax.effects_barrier()
            elapsed[f"{module.__name__.removeprefix('raytrax.')}.{name}"] = time.perf_counter() - start
            return result

        return timing

    for module, name, function in original:
        setattr(module, name, wrap(module, name, function))
    try:
        yield elapsed
    finally:
        for module, name, function in original:
            setattr(module, name, function)


def build(name: str, kappa: float) -> tuple[Problem, dict[str, float]]:
    """Read one case, timing every host-side stage it passes through."""
    with timed(BUILD_STAGES) as elapsed:
        case = prepare(name, kappa)
    return case, elapsed


def run(
    case: Problem, num_rays: int, lanes: int | None, max_steps: int, repeats: int, refill_interval: int
) -> dict[str, Any]:
    """Time the compile and the marches of one `trace`, at the width it would pick."""
    key = jax.random.key(0)
    lanes = (lanes or tuning.device_width()) or num_rays

    def trace() -> Any:
        return transport.trace(
            key, case, num_rays=num_rays, lanes=lanes, max_steps=max_steps, refill_interval=refill_interval
        )

    start = time.perf_counter()
    tally = jax.block_until_ready(trace())
    deposit = tally.estimate(case)
    first = time.perf_counter() - start

    runs = []
    for _ in range(repeats):
        start = time.perf_counter()
        jax.block_until_ready(trace())
        runs.append(time.perf_counter() - start)
    median = statistics.median(runs)

    emitted = float(case.sources.total)
    return {
        "lanes": lanes,
        "num_rays": num_rays,
        # The first call is compile plus one run; the difference is the compile.
        "compile_s": first - median,
        "median_s": median,
        "min_s": min(runs),
        "alive_at_end": int(tally.rays.alive.sum()),
        "deposited_fraction": float(deposit.cells.sum() + deposit.faces.sum() + deposit.escaped) / emitted,
        "trace": trace,
    }


def capture(trace: Any, out: Path) -> Path:
    """Trace one warmed run, and return the xplane file it wrote."""
    with jax.profiler.trace(str(out)):
        jax.block_until_ready(trace())
    planes = sorted(out.rglob("*.xplane.pb"))
    if not planes:
        raise FileNotFoundError(f"no xplane written under {out}")
    return planes[-1]


def steps_taken(rows: list[dict[str, Any]]) -> int:
    """How many loop iterations the march ran, from the kernel launch counts.

    Read off the kernel that spends the most time, which is in the loop body and
    so launches once per iteration. A count over all kernels does not work: the
    setup launches many kernels once, and an RNG split or the refill branch
    launches its own several times per step.
    """
    if not rows:
        return 0
    return int(max(rows, key=lambda row: row["total_duration_us"])["occurrences"])


HEADER = f"{'kernel':<44}{'calls':>7}{'us/call':>10}{'total us':>11}{'share':>8}{'reg':>6}{'occ%':>7}"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--case", default="reentry", choices=sorted(CASES))
    parser.add_argument("--rays", type=int, default=1_000_000)
    parser.add_argument("--lanes", type=int, default=None, help="override the width `tuning.device_width` picks")
    parser.add_argument("--max-steps", type=int, default=2_000)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--refill-interval", type=int, default=8, help="steps between refills of the dead lanes")
    parser.add_argument("--kappa", type=float, default=2.0)
    parser.add_argument("--top", type=int, default=12)
    parser.add_argument("--out", type=Path, default=Path("profile"))
    parser.add_argument("--x32", action="store_true", help="undo the package's x64 default, to price fp64")
    parser.add_argument("--json", type=Path, default=None)
    args = parser.parse_args()

    if args.x32:
        jax.config.update("jax_enable_x64", False)

    env = environment()
    print(f"# {env['device_kind']} ({env['platform']}) | jax {env['jax']} | x64={env['x64']}")

    case, host = build(args.case, args.kappa)
    record = run(case, args.rays, args.lanes, args.max_steps, args.repeats, args.refill_interval)
    plane = capture(record.pop("trace"), args.out)
    rows = kernel_rows(plane)
    device_us = sum(row["total_duration_us"] for row in rows)
    steps = steps_taken(rows)

    print(
        f"# {args.case}: cells={case.scene.num_cells} faces={case.scene.num_faces} "
        f"rays={args.rays} lanes={record['lanes']} kappa={args.kappa}"
    )
    print(f"{'stage':<28}{'seconds':>10}{'share':>8}")
    total = sum(host.values()) + record["compile_s"] + record["median_s"]
    staged = [*host.items(), ("compile", record["compile_s"]), ("march", record["median_s"])]
    for stage, seconds in staged:
        print(f"{stage:<28}{seconds:>10.3f}{seconds / total * 100:>7.1f}%")
    print(f"{'total':<28}{total:>10.3f}")
    lane_steps = steps * record["lanes"]
    print(
        f"# {steps} steps x {record['lanes']} lanes = {lane_steps / 1e6:.1f}M lane-steps at "
        f"{lane_steps / record['median_s'] / 1e6:.1f} Mlanestep/s, {device_us / 1e3:.1f} ms on device "
        f"({device_us / 1e3 / record['median_s'] / 10:.0f}% of the march)"
    )
    print(f"# {record['deposited_fraction']:.6f} of the source deposited, {record['alive_at_end']} alive at the cap")

    print(HEADER)
    for row in rows[: args.top]:
        print(
            f"{row['kernel_name']:<44}{row['occurrences']:>7.0f}{row['avg_duration_us']:>10.1f}"
            f"{row['total_duration_us']:>11.0f}{row['total_duration_us'] / device_us * 100:>7.1f}%"
            f"{row['registers_per_thread']:>6.0f}{row['occupancy_pct']:>7.1f}"
        )

    print(f"# trace in {args.out}; open with `uv run --group profiling xprof --logdir {args.out}`")

    if args.json:
        payload = {
            "environment": env,
            "case": args.case,
            "kappa": args.kappa,
            "host_s": host,
            "run": record,
            "steps": steps,
            "total_device_us": device_us,
            "kernels": rows,
        }
        args.json.write_text(json.dumps(payload, indent=2) + "\n")
        print(f"# wrote {args.json}")
