"""Time `transport.march` over a mesh, lane and path ladder.

    uv run python scripts/benchmark.py --case slab_planar --lanes 65536
    uv run python scripts/benchmark.py --ladder --json bench.json

Reports compile time and run time separately, because `march` carries no
`static_argnums` and takes its statics implicitly, so every distinct
(case, lanes, steps) is a fresh compile.

Timing follows the JAX guide: warm up once, then block on the whole returned
pytree, not one array, or the numbers measure async dispatch instead of compute.

Set `JAX_PLATFORMS=cpu` to run the same ladder on the host. Meshes come from
`examples/meshes/generate.py` and are not committed.
"""

import argparse
import json
import platform
import statistics
import subprocess
import time
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from raytrax import interactions, sources, transport
from raytrax import scene as scene_module
from raytrax.interactions import Kind
from raytrax.io import vtk
from raytrax.mesh import geometry
from raytrax.paths import AxisymmetricPath, LinearPath
from raytrax.problem import Problem

MESHES = Path(__file__).resolve().parent.parent / "examples" / "meshes"

# Mesh file, and whether it is read as the meridional half-plane. Mirrors
# `prepare` in examples/convergence.py, which is a marimo app and cannot be
# imported; the two must agree or the benchmark stops describing the study.
CASES = {
    "slab_planar": ("slab", False),
    "slab_axisymmetric": ("slab", True),
    "cylinder_coarse": ("cylinder_coarse", False),
    "plane": ("plane", False),
    "cylinder": ("cylinder", False),
    "reentry": ("reentry", True),
}


def prepare(name: str, kappa: float = 1.0) -> Problem:
    """Read one case into the problem to simulate."""
    stem, axisymmetric = CASES[name]
    path = MESHES / f"{stem}.msh"
    if not path.exists():
        raise FileNotFoundError(f"{path} missing - run `uv run python examples/meshes/generate.py`")

    mesh = vtk.read(path)
    geom = geometry.revolve(mesh.geometry) if axisymmetric else mesh.geometry
    trace_path = AxisymmetricPath() if axisymmetric else LinearPath()
    scene = scene_module.build(mesh.topology, geom, mesh.groups)

    if stem == "reentry":
        kinds = {"shock": Kind.ABSORB, "wall": Kind.ABSORB, "outflow": Kind.ESCAPE, "axis": Kind.ABSORB}
        emitting = "shock"
    else:
        kinds = {"hot_end": Kind.ABSORB, "wall": Kind.ABSORB, "lateral": Kind.SPECULAR}
        if "axis" in mesh.groups.names:
            # Revolved, the y = 0 edge is the axis; extruded it is just another
            # side and has to match `lateral` or the slab stops being symmetric.
            kinds["axis"] = Kind.ABSORB if axisymmetric else Kind.SPECULAR
        emitting = "hot_end"
    table = interactions.build(mesh.groups, kinds)

    faces = np.flatnonzero(mesh.groups.face_group == mesh.groups.index(emitting))
    source = sources.build(jnp.zeros(mesh.topology.num_cells), geom.faces.area[faces], faces)
    return Problem(scene, trace_path, table, jnp.full(mesh.topology.num_cells, kappa), source)


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


def environment() -> dict[str, Any]:
    """Enough about this machine to compare a run against another one."""
    device = jax.local_devices()[0]
    try:
        driver = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        ).stdout.strip()
    except OSError, subprocess.SubprocessError:
        driver = ""
    return {
        "device_kind": device.device_kind,
        "platform": device.platform,
        "driver": driver,
        "host": platform.node(),
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

    case = prepare(args.case, args.kappa)
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
