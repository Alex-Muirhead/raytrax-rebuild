import marimo

__generated_with = "0.24.0"
app = marimo.App(width="medium")

with app.setup:
    from pathlib import Path

    import marimo as mo
    import numpy as np

    from raytrax_examples import benchmark, export
    from raytrax_examples.cases import cylinder, reentry, slab
    from raytrax_examples.typst import figure

    HERE = Path(__file__).parent
    DATA = HERE / "data"
    FIGURES = HERE / "figures.typ"
    DATA.mkdir(exist_ok=True)

    COLUMNS = ("lanes", "compile_s", "median_s", "min_s", "ms_per_step", "lane_steps_per_s", "peak_mib", "alive_at_end")


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


if __name__ == "__main__":
    app.run()
