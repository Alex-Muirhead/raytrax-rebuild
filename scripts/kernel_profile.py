"""Break `transport.march` down by CUDA kernel, using xprof.

    uv run --group profiling python scripts/kernel_profile.py --case slab_planar
    uv run --group profiling python scripts/kernel_profile.py --lanes 65536 --json kernels.json

Captures a trace of a warmed march and reports device time per kernel, so a
cost is measured rather than inferred by differencing whole-run timings.

XLA fuses the entire `while` body, so the HLO-level tools report it as one
opaque op and attribute nothing inside it. The kernel table is the level that
still resolves, and `--attribute` names each kernel by the `jax.named_scope`s
fused into it, then rolls the kernel times up per scope. XLA keeps no source
file or line, so those scopes are the only way back to source.

Needs the `profiling` dependency group.
"""

import argparse
import collections
import json
import re
from pathlib import Path
from typing import Any

import jax
from raytrax_examples.benchmark import make_rays
from raytrax_examples.cases import CASES, prepare

from raytrax import transport
from raytrax.problem import Problem

# `  %loop_multiply_fusion.1 = f64[...] fusion(...), kind=kLoop, calls=%fused_multiply.1, ...`
FUSION = re.compile(r"^\s+%(\S+) = .*\bfusion\(.*\bcalls=%([^,\s]+)")
OP_NAME = re.compile(r'op_name="([^"]*)"')

# Every op inside the loop carries this, and it says nothing about which op.
# `scan` adds `eval_jaxpr`, and the refilling loop nests the march in its own body.
TRACED_PREFIX = re.compile(r"^(?:(?:jit\([^)]*\)|while/body|eval_jaxpr)/)*")

# `vmap(crossing)` is one of our `jax.named_scope`s, batched; `vmap()` and
# `jit(nanmin)` are a transform and a jnp helper, and name nothing of ours.
SCOPE = re.compile(r"^(?:vmap|jit)\((?P<name>[^)]*)\)$|^(?P<plain>[^()]+)$")
HELPERS = frozenset({"nanmin", "_nanargmin", "_where", "_compare"})


def scope_of(op_name: str) -> str:
    """The `named_scope` path an operation sits in, or `-` outside them all."""
    parts = TRACED_PREFIX.sub("", op_name).split("/")[:-1]
    named = []
    for part in parts:
        match = SCOPE.match(part)
        if match:
            name = match.group("name") or match.group("plain")
            if name and name not in HELPERS:
                named.append(name)
    return "/".join(named) or "-"


def compiled_text(case: Problem, rays: transport.Rays, max_steps: int) -> str:
    """The optimised HLO for the same march the trace captures."""
    # pyrefly: ignore[missing-attribute] - it does not model what filter_jit returns
    lowered = transport.march.lower(jax.random.key(1), rays, case, steps=max_steps)
    # equinox wraps the jax Compiled; the analyses live on the inner one.
    return lowered.compile().compiled.as_text()


def capture(case: Problem, rays: transport.Rays, max_steps: int, out: Path) -> Path:
    """Trace one warmed march, and return the xplane file it wrote."""

    def march():
        return transport.march(jax.random.key(1), rays, case, steps=max_steps)

    jax.block_until_ready(march())  # warm, so the trace holds no compile
    with jax.profiler.trace(str(out)):
        jax.block_until_ready(march())

    planes = sorted(out.rglob("*.xplane.pb"))
    if not planes:
        raise FileNotFoundError(f"no xplane written under {out}")
    return planes[-1]


def kernel_rows(plane: Path) -> list[dict[str, Any]]:
    """Per-kernel device time from the captured trace."""
    from xprof.convert import raw_to_tool_data

    raw, _ = raw_to_tool_data.xspace_to_tool_data([str(plane)], "kernel_stats", {})
    table = json.loads(raw)
    table = table[0] if isinstance(table, list) else table
    columns = [column["id"] for column in table["cols"]]
    return [dict(zip(columns, [cell.get("v") for cell in row["c"]], strict=True)) for row in table["rows"]]


def fused_operations(text: str, by_scope: bool = True) -> dict[str, collections.Counter[str]]:
    """For each fusion kernel, what XLA folded into it, counted by `named_scope` or by operation.

    `text` is the optimised HLO. `by_scope` names each operation by the scope it
    came from rather than by the primitive it is.

    Keyed the way the profiler names kernels, which is the HLO instruction name
    with `.` replaced by `_`.
    """
    bodies: dict[str, list[str]] = {}
    current = None
    for line in text.splitlines():
        if line.startswith("%") and "(" in line:
            current = line[1:].split(" ", 1)[0]
            bodies[current] = []
        elif current is not None:
            bodies[current].append(line)

    calls = {}
    for line in text.splitlines():
        match = FUSION.match(line)
        if match:
            calls[match.group(1).replace(".", "_")] = match.group(2)

    attributed = {}
    for kernel, body in calls.items():
        counts = collections.Counter[str]()
        for line in bodies.get(body, []):
            for name in OP_NAME.findall(line):
                counts[scope_of(name) if by_scope else TRACED_PREFIX.sub("", name)] += 1
        attributed[kernel] = counts
    return attributed


def label(counts: collections.Counter[str], width: int = 3) -> str:
    """The operations that dominate one kernel, most frequent first."""
    if not counts:
        return "-"
    return ", ".join(f"{name} x{n}" for name, n in counts.most_common(width))


def scope_time(rows: list[dict[str, Any]], fused: dict[str, collections.Counter[str]]) -> collections.Counter[str]:
    """Device microseconds per `named_scope`, over every kernel.

    A kernel holds ops from several scopes at once, and the profiler times the
    kernel and not the op, so a kernel's time is split between its scopes in
    proportion to how many of its operations each contributed. That is a share
    of instructions, not of cycles: read it as which scope to look at, not as
    what that scope costs.
    """
    totals = collections.Counter[str]()
    for row in rows:
        counts = fused.get(row["kernel_name"])
        if not counts:
            totals["-"] += row["total_duration_us"]
            continue
        operations = sum(counts.values())
        for name, count in counts.items():
            totals[name] += row["total_duration_us"] * count / operations
    return totals


HEADER = f"{'kernel':<44}{'calls':>7}{'us/call':>10}{'total us':>11}{'share':>8}{'reg':>6}{'occ%':>7}"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--case", default="slab_planar", choices=sorted(CASES))
    parser.add_argument("--lanes", type=int, default=65_536)
    parser.add_argument("--max-steps", type=int, default=400)
    parser.add_argument("--top", type=int, default=12)
    parser.add_argument("--attribute", action="store_true", help="name each kernel by the scopes fused into it")
    parser.add_argument("--by-operation", action="store_true", help="attribute to jaxpr primitives, not scopes")
    parser.add_argument("--out", type=Path, default=Path("profile"))
    parser.add_argument("--x32", action="store_true", help="undo the package's x64 default, to price fp64")
    parser.add_argument("--json", type=Path, default=None)
    args = parser.parse_args()

    if args.x32:
        jax.config.update("jax_enable_x64", False)

    case = prepare(args.case).problem
    rays = make_rays(case, args.lanes, 0)
    text = compiled_text(case, rays, args.max_steps)
    plane = capture(case, rays, args.max_steps, args.out)

    rows = kernel_rows(plane)
    total = sum(row["total_duration_us"] for row in rows)
    fused = fused_operations(text, by_scope=not args.by_operation) if args.attribute else {}

    print(f"# {args.case} lanes={args.lanes} max_steps={args.max_steps} x64={bool(jax.config.jax_enable_x64)}")
    print(f"# {len(rows)} kernels, {total / 1e3:.1f} ms on device, {total / args.max_steps:.1f} us per step")
    print(HEADER)
    for row in rows[: args.top]:
        share = row["total_duration_us"] / total * 100
        print(
            f"{row['kernel_name']:<44}{row['occurrences']:>7.0f}{row['avg_duration_us']:>10.1f}"
            f"{row['total_duration_us']:>11.0f}{share:>7.1f}%{row['registers_per_thread']:>6.0f}"
            f"{row['occupancy_pct']:>7.1f}"
        )
        if args.attribute:
            print(f"    {label(fused.get(row['kernel_name'], collections.Counter()))}")

    if args.attribute:
        print(f"{'scope':<44}{'total us':>11}{'share':>8}")
        for name, microseconds in scope_time(rows, fused).most_common():
            print(f"{name:<44}{microseconds:>11.0f}{microseconds / total * 100:>7.1f}%")

    if args.json:
        payload = {
            "case": args.case,
            "lanes": args.lanes,
            "max_steps": args.max_steps,
            "x64": bool(jax.config.jax_enable_x64),
            "total_device_us": total,
            "kernels": [row | {"fused_ops": dict(fused.get(row["kernel_name"], {}))} for row in rows],
        }
        args.json.write_text(json.dumps(payload, indent=2) + "\n")
        print(f"# wrote {args.json}")
