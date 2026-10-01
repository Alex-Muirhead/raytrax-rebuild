import marimo

__generated_with = "0.24.0"
app = marimo.App(width="medium")

with app.setup:
    from pathlib import Path

    import jax
    import marimo as mo
    import numpy as np

    from raytrax import jit_tools, transport
    from raytrax_examples import export, profiles
    from raytrax_examples.cases import cylinder, slab
    from raytrax_examples.typst import figure

    HERE = Path(__file__).parent
    DATA = HERE / "data"
    FIGURES = HERE / "figures.typ"
    DATA.mkdir(exist_ok=True)

    RAY_COUNTS = (1_000, 10_000, 100_000, 1_000_000, 10_000_000)
    NUM_SEEDS = 5


@app.cell(hide_code=True)
def _():
    mo.md("""
    # Sampling convergence

    The tangent-slab verification shows the profile comes out right at one ray
    count; it does not say how the error falls with the number of rays, nor
    where it stops falling because the mesh has become the limit. This sweeps
    the ray count over four decades, five seeds each.

    Three cases, the same physics as the verification: a hot wall at low x
    radiating into cold absorbing gas, lateral boundary mirrored so the domain
    is an infinite slab.

    - **slab_planar** and **slab_axisymmetric** - one jumbled 2D mesh read two
      ways. Same cells either way, so the two curves must agree.
    - **cylinder_coarse** - 3D tets, coarse enough that the mesh limit is easy to
      see.

    Two errors are measured against the same closed form, both divided by the
    mean heating the domain would have had if every joule landed uniformly.

    - **per cell** - volume-weighted RMS over every cell, against the point value
      `2 k E2(k x)`. Every cell carries its own noise, so this starts high, but a
      cell's volume-weighted mean x *is* its centroid, so the linear part of the
      profile cancels exactly and the floor left over is second order in cell
      size.
    - **binned** - RMS over the 25 axial bins of the verification profile,
      against the cell-averaged `2 (E3(ka) - E3(kb)) / (b - a)`. Averaging cells
      first starts this an order of magnitude lower, but cells are sorted into
      bins by centroid and do not tile them, so the volume in a bin is not the
      interval the closed form averages over. Its centre sits off the bin
      midpoint, by about 2% of a bin on the slab and 15% on the coarse cylinder,
      and that misalignment is first order.

    First order beats second order, so the binned floor is the higher of the two
    and is reached sooner, on the slab as much as on the cylinder. At kappa = 1
    the floors are 1.0e-3 and 2.5e-3 on the slab mesh, 9.8e-3 and 2.5e-2 on the
    cylinder. Neither is a property of the rays.

    Run as a script, the sweep always runs; `-- --largest 100000` caps the ray
    count for a quick check.
    """)
    return


@app.cell
def _():
    largest_default = mo.cli_args().get("largest") or f"{RAY_COUNTS[-1]}"
    kappa = mo.ui.slider(0.1, 5.0, 0.1, value=1.0, label="Absorptivity kappa")
    largest = mo.ui.dropdown(
        [f"{count}" for count in RAY_COUNTS], value=str(largest_default), label="Largest ray count"
    )
    running = mo.ui.checkbox(mo.app_meta().mode == "script", label="Run the sweep")
    mo.hstack([kappa, largest, running], justify="start", gap=2)
    return kappa, largest, running


@app.function
def run(case, rays, key, lanes):
    """Trace one case at one ray count, with the two fractions that police it.

    `max_steps` is this notebook's whole cost, not a safety net: a couple of
    mirrored rays bounce long after they carry anything, the batch pays for its
    longest, so the march runs the cap out every time. By 2,000 a full batch has
    drained to 1e-15 of the emitted energy and a refilled one has spent a
    ten-fold budget over; `spent` and `flight` are what say so on any other case.
    """
    problem = case.problem
    tally = transport.trace(key, problem, num_rays=rays, lanes=lanes, max_steps=2_000)
    deposit = tally.estimate(problem)
    escaped = float(deposit.escaped)
    flight = float(tally.in_flight(problem))
    spent = float(deposit.cells.sum() + deposit.faces.sum()) + escaped + flight

    total = float(problem.sources.total)
    heating = np.asarray(deposit.cells) / case.geom.cells.volume
    return heating, spent / total, flight / total


@app.cell
def _(kappa, running):
    mo.stop(
        not running.value,
        mo.md("*Tick **Run the sweep** above. A full sweep is 75 traces, about six minutes on a GPU.*"),
    )
    slab_planar = slab.case(kappa=kappa.value)
    slab_axisymmetric = slab.case(axisymmetric=True, kappa=kappa.value)
    cylinder_coarse = cylinder.case(kappa=kappa.value, mesh_size=0.6)
    return cylinder_coarse, slab_axisymmetric, slab_planar


@app.function
def sweep(case, kappa, counts, keys, lanes, bar):
    """Trace one case at every ray count and seed, and give one row per trace.

    Each row is the ray count, the seed, the per-cell and binned errors, the
    share of the launch budget spent, and the share of energy left in flight.
    """
    rows = []
    for count in counts:
        for seed, key in enumerate(keys):
            heating, spent, flight = run(case, count, key, lanes)
            rows.append(
                (
                    count,
                    seed,
                    profiles.cell_error(case, heating, kappa),
                    profiles.profile_error(case, heating, kappa),
                    spent,
                    flight,
                )
            )
            bar.update()
    return np.array(rows)


@app.cell
def _(cylinder_coarse, kappa, largest, slab_axisymmetric, slab_planar):
    counts = [count for count in RAY_COUNTS if count <= int(largest.value)]
    keys = jax.random.split(jax.random.key(0), NUM_SEEDS)
    # One width for every ray count, so each curve is a statement about rays
    # rather than about batches.
    lanes = jit_tools.device_width()
    with mo.status.progress_bar(total=3 * len(counts) * NUM_SEEDS) as bar:
        rows_planar = sweep(slab_planar, kappa.value, counts, keys, lanes, bar)
        rows_axisymmetric = sweep(slab_axisymmetric, kappa.value, counts, keys, lanes, bar)
        rows_coarse = sweep(cylinder_coarse, kappa.value, counts, keys, lanes, bar)
    return rows_axisymmetric, rows_coarse, rows_planar


@app.function
def aggregate(rows):
    """Collapse the per-seed rows to mean, min and max per ray count."""
    counts = np.unique(rows[:, 0])
    groups = [rows[rows[:, 0] == count] for count in counts]
    columns = [counts]
    for column in (2, 3):
        values = [group[:, column] for group in groups]
        columns += [np.array([reduce(value) for value in values]) for reduce in (np.mean, np.min, np.max)]

    # An N^-1/2 guide for each metric, anchored on the smallest ray count.
    columns += [mean[0] * np.sqrt(counts[0] / counts) for mean in (columns[1], columns[4])]
    return columns


@app.cell
def _(rows_axisymmetric, rows_coarse, rows_planar):
    # Unspent budget means rays that never launched; energy still in flight
    # means rays that never finished. Neither shows up in the errors as
    # anything but a floor, so they are called out rather than plotted.
    _all = np.vstack([rows_planar, rows_axisymmetric, rows_coarse])
    _unspent = abs(_all[:, 4] - 1).max()
    _flight = _all[:, 5].max()
    _note = f"Worst unspent launch budget {_unspent:.1e}, worst energy left in flight {_flight:.1e}"
    if _unspent > 1e-9 or _flight > 1e-6:
        _note += " - **too large**: raise `max_steps` in `run`"
    mo.md(_note + ".")
    return


@app.cell
def _(rows_axisymmetric, rows_coarse, rows_planar):
    _header = "rays,cell_rmse,cell_lo,cell_hi,bin_rmse,bin_lo,bin_hi,cell_sqrt_law,bin_sqrt_law"
    export.save_csv(DATA / "slab_planar_rmse.csv", aggregate(rows_planar), _header)
    export.save_csv(DATA / "slab_axisymmetric_rmse.csv", aggregate(rows_axisymmetric), _header)
    export.save_csv(DATA / "cylinder_coarse_rmse.csv", aggregate(rows_coarse), _header)

    # Every trace, for anything the figure does not show.
    _header = "rays,seed,cell_rmse,bin_rmse,spent,in_flight"
    export.save_csv(DATA / "slab_planar_seeds.csv", rows_planar.T, _header)
    export.save_csv(DATA / "slab_axisymmetric_seeds.csv", rows_axisymmetric.T, _header)
    export.save_csv(DATA / "cylinder_coarse_seeds.csv", rows_coarse.T, _header)

    mo.image(figure(FIGURES, "rmse"))
    return


if __name__ == "__main__":
    app.run()
