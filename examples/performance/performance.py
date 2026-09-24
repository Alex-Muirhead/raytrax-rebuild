import marimo

__generated_with = "0.24.0"
app = marimo.App(width="medium")

with app.setup:
    from pathlib import Path

    import marimo as mo
    import numpy as np

    from raytrax_examples import benchmark, cases, export
    from raytrax_examples.typst import compile_figures

    HERE = Path(__file__).parent
    DATA = HERE / "data"

    CASES = ("slab_planar", "slab_axisymmetric", "cylinder", "reentry")
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
def _(powers, repeats, running, steps):
    def _ladder():
        if not running.value:
            return None
        low, high = powers.value
        lanes = [1 << power for power in range(low, high + 1)]
        results = {}
        with mo.status.progress_bar(total=len(CASES) * len(lanes)) as bar:
            for name in CASES:
                problem = cases.prepare(name).problem
                records = []
                for count in lanes:
                    try:
                        records.append(benchmark.measure(problem, count, steps.value, repeats.value, 0, 1e-8))
                    except Exception:  # noqa: BLE001 - an out-of-memory ends this case's ladder, not the run
                        break
                    finally:
                        bar.update()
                results[name] = records
        return results

    ladder = _ladder()
    return (ladder,)


@app.cell(hide_code=True)
def _(ladder):
    def _table():
        if ladder is None:
            return mo.md("*Tick **Run the ladder** above.*")
        lines = []
        for name, records in ladder.items():
            lines += [f"**{name}**", "```", benchmark.HEADER, *map(benchmark.row, records), "```", ""]
        return mo.md("\n".join(lines))

    _table()
    return


@app.cell(hide_code=True)
def _(ladder):
    def _export():
        if ladder is None:
            return []
        written = [export.save_json(DATA / "environment.json", benchmark.environment())]
        for name, records in ladder.items():
            # A backend that reports no peak memory gives None; the CSV takes NaN.
            columns = [np.array([np.nan if r[c] is None else r[c] for r in records], dtype=float) for c in COLUMNS]
            written.append(export.save_csv(DATA / f"{name}.csv", columns, ",".join(COLUMNS)))
        return written + compile_figures(HERE / "figures.typ")

    written = _export()
    return (written,)


@app.cell(hide_code=True)
def _(written):
    mo.vstack([mo.image(path) for path in written if path.parent.name == "figures"])
    return


if __name__ == "__main__":
    app.run()
