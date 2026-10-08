import marimo

__generated_with = "0.24.0"
app = marimo.App(width="medium")

with app.setup:
    from pathlib import Path

    import jax
    import marimo as mo
    import numpy as np

    from raytrax import jit_tools
    from raytrax_examples import benchmark, export
    from raytrax_examples.cases import cylinder, reentry, slab
    from raytrax_examples.typst import figure

    HERE = Path(__file__).parent
    DATA = HERE / "data"
    FIGURES = HERE / "figures.typ"
    DATA.mkdir(exist_ok=True)

    COLUMNS = ("lanes", "compile_s", "median_s", "min_s", "ms_per_step", "lane_steps_per_s", "peak_mib", "alive_at_end")
    TRACE_COLUMNS = ("rays", "lanes", "compile_s", "median_s", "min_s", "max_s", "rays_per_s", "energy_error")

    # Half-decades of rays, from 1e2 to 1e7.
    RAY_COUNTS = [round(10 ** (power / 2)) for power in range(4, 15)]

    # The CPU march keeps the same lane-steps per second from about 1024 lanes
    # up, so a wider batch only adds dead lanes at the end of a run.
    CPU_LANES = 4096


@app.cell(hide_code=True)
def _():
    mo.md("""
    # Performance

    Throughput of `transport.march` against the number of lanes marched
    together, on the 2D planar, 2D axisymmetric, 3D and re-entry meshes. Each
    point compiles once, then times a fixed number of steps several times and
    keeps the median, blocking on the whole result so that the time is compute
    and not asynchronous dispatch.

    A march with more lanes than the device has threads to run them gains no
    parallelism, so throughput rises with lanes until the device is full and
    then flattens. Compile time is reported separately: every distinct lane
    count is a fresh compile.

    Run as a script, the ladder always runs; `-- --to-power 16` stops it early
    for a quick check. `python -m raytrax_examples.benchmark` times one case at
    one lane count from the command line.
    """)
    return


@app.cell
def _():
    to_power = int(str(mo.cli_args().get("to-power") or 22))
    powers = mo.ui.range_slider(10, 24, 1, value=(13, to_power), label="Lanes, as powers of two")
    steps = mo.ui.number(50, 2000, 50, value=400, label="Steps")
    repeats = mo.ui.number(1, 20, 1, value=5, label="Repeats")
    running = mo.ui.checkbox(mo.app_meta().mode == "script", label="Run the ladder")
    mo.hstack([powers, steps, repeats, running], justify="start", gap=2)
    return powers, repeats, running, steps


@app.cell
def _(running):
    mo.stop(not running.value, mo.md("*Tick **Run the ladder** above.*"))
    slab_planar = slab.case().problem
    slab_axisymmetric = slab.case(axisymmetric=True).problem
    cylinder_fine = cylinder.case().problem
    capsule = reentry.case().problem
    return capsule, cylinder_fine, slab_axisymmetric, slab_planar


@app.function
def ladder(problem, lanes, steps, repeats, bar):
    """Time one problem at each lane count, and give one record per count that fits on the device."""
    records = []
    for count in lanes:
        try:
            records.append(benchmark.measure(problem, count, steps, repeats, 0, 1e-8))
        except Exception:  # noqa: BLE001 - an out-of-memory ends this ladder, not the run
            bar.update(len(lanes) - len(records))
            break
        bar.update()
    return records


@app.cell
def _(capsule, cylinder_fine, powers, repeats, slab_axisymmetric, slab_planar, steps):
    low, high = powers.value
    lanes = [1 << power for power in range(low, high + 1)]
    with mo.status.progress_bar(total=4 * len(lanes)) as bar:
        records_planar = ladder(slab_planar, lanes, steps.value, repeats.value, bar)
        records_axisymmetric = ladder(slab_axisymmetric, lanes, steps.value, repeats.value, bar)
        records_cylinder = ladder(cylinder_fine, lanes, steps.value, repeats.value, bar)
        records_reentry = ladder(capsule, lanes, steps.value, repeats.value, bar)
    return records_axisymmetric, records_cylinder, records_planar, records_reentry


@app.cell
def _(records_axisymmetric, records_cylinder, records_planar, records_reentry):
    mo.md(
        "\n".join(
            [
                f"**{name}**\n```\n{benchmark.HEADER}\n" + "\n".join(map(benchmark.row, records)) + "\n```"
                for name, records in (
                    ("slab_planar", records_planar),
                    ("slab_axisymmetric", records_axisymmetric),
                    ("cylinder", records_cylinder),
                    ("reentry", records_reentry),
                )
            ]
        )
    )
    return


@app.function
def columns(records):
    """Give the records as one float array per column of `COLUMNS`, with NaN where a backend reports no peak memory."""
    return [np.array([np.nan if r[c] is None else r[c] for r in records], dtype=float) for c in COLUMNS]


@app.cell
def _(records_axisymmetric, records_cylinder, records_planar, records_reentry):
    export.save_json(DATA / "environment.json", benchmark.environment())
    _header = ",".join(COLUMNS)
    export.save_csv(DATA / "slab_planar.csv", columns(records_planar), _header)
    export.save_csv(DATA / "slab_axisymmetric.csv", columns(records_axisymmetric), _header)
    export.save_csv(DATA / "cylinder.csv", columns(records_cylinder), _header)
    export.save_csv(DATA / "reentry.csv", columns(records_reentry), _header)

    mo.hstack([mo.image(figure(FIGURES, "throughput")), mo.image(figure(FIGURES, "compile-time"))])
    return


@app.cell(hide_code=True)
def _():
    mo.md("""
    ## GPU against CPU

    The time of a whole `transport.trace` on the 2D planar slab, against the
    number of rays, on the GPU and on the host CPU in the same process. Each
    point compiles once, then times five runs and keeps the median; the
    compile time is kept separately.

    A small run is all latency: each chunk of steps on the GPU costs a fixed
    launch and synchronisation time, so its time has a floor well above the
    CPU's. A large run is all throughput, which the GPU has more of. The
    crossover is the ray count above which the GPU stays faster; near it the
    two times are close, and timing noise can cross them more than once.

    The GPU marches `jit_tools.device_width()` lanes and the CPU `CPU_LANES`,
    each capped at the ray count. Run as a script, the sweep always runs;
    `-- --largest 100000` caps the ray count.
    """)
    return


@app.cell
def _():
    largest_default = mo.cli_args().get("largest") or f"{RAY_COUNTS[-1]}"
    largest = mo.ui.dropdown(
        [f"{count}" for count in RAY_COUNTS], value=str(largest_default), label="Largest ray count"
    )
    crossing = mo.ui.checkbox(mo.app_meta().mode == "script", label="Run the crossover")
    mo.hstack([largest, crossing], justify="start", gap=2)
    return crossing, largest


@app.function
def sweep(problem, device, width, counts, bar):
    """Time whole traces of one problem on one device, and give one record per ray count."""
    records = []
    with jax.default_device(device):
        placed = jax.device_put(problem, device)
        # The first run on a device also pays to start it, which no later run does.
        benchmark.measure_trace(placed, counts[0], counts[0], 1, 0)
        for count in counts:
            records.append(benchmark.measure_trace(placed, count, min(count, width), 5, 0))
            bar.update()
    return records


@app.function
def crossover(rays, gpu_s, cpu_s):
    """Give the ray count above which the GPU stays faster than the CPU, or None if no such count is in the range."""
    log_ratio = np.log(gpu_s / cpu_s)
    slower = np.flatnonzero(log_ratio >= 0)
    if not slower.size or slower[-1] == len(rays) - 1:
        return None
    i = slower[-1]
    # The ratio falls through zero between i and i + 1; np.interp wants it rising.
    return float(np.exp(np.interp(0.0, log_ratio[i : i + 2][::-1], np.log(rays[i : i + 2])[::-1])))


@app.cell
def _(crossing, largest):
    mo.stop(not crossing.value, mo.md("*Tick **Run the crossover** above.*"))
    counts = [count for count in RAY_COUNTS if count <= int(largest.value)]
    gpu, cpu = jax.devices("gpu")[0], jax.devices("cpu")[0]
    gpu_lanes = jit_tools.device_width() or 1 << 16
    slab_trace = slab.case().problem
    with mo.status.progress_bar(total=2 * len(counts)) as _bar:
        records_gpu = sweep(slab_trace, gpu, gpu_lanes, counts, _bar)
        records_cpu = sweep(slab_trace, cpu, CPU_LANES, counts, _bar)
    return cpu, gpu, gpu_lanes, records_cpu, records_gpu


@app.cell
def _(cpu, gpu, gpu_lanes, records_cpu, records_gpu):
    _gpu = [np.array([r[c] for r in records_gpu], dtype=float) for c in TRACE_COLUMNS]
    _cpu = [np.array([r[c] for r in records_cpu], dtype=float) for c in TRACE_COLUMNS]
    _header = ",".join(TRACE_COLUMNS)
    export.save_csv(DATA / "backend_gpu.csv", _gpu, _header)
    export.save_csv(DATA / "backend_cpu.csv", _cpu, _header)

    _median = TRACE_COLUMNS.index("median_s")
    crossover_rays = crossover(_gpu[0], _gpu[_median], _cpu[_median])
    export.save_json(
        DATA / "crossover.json",
        {
            "rays": crossover_rays,
            "gpu_lanes": gpu_lanes,
            "cpu_lanes": CPU_LANES,
            "gpu": benchmark.environment(gpu),
            "cpu": benchmark.environment(cpu),
        },
    )

    _found = "no crossover in this range" if crossover_rays is None else f"crossover at about {crossover_rays:.3g} rays"
    _table = "\n".join(
        f"{g['rays']:>9}{g['median_s']:>11.4f}{c['median_s']:>11.4f}{c['median_s'] / g['median_s']:>9.2f}"
        for g, c in zip(records_gpu, records_cpu, strict=True)
    )
    mo.vstack(
        [
            mo.md(f"**{_found}**\n```\n{'rays':>9}{'gpu_s':>11}{'cpu_s':>11}{'speedup':>9}\n{_table}\n```"),
            mo.image(figure(FIGURES, "crossover")),
        ]
    )
    return


if __name__ == "__main__":
    app.run()
