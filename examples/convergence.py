import marimo

__generated_with = "0.24.0"
app = marimo.App(width="medium")

with app.setup:
    from pathlib import Path
    from typing import NamedTuple

    import jax
    import jax.numpy as jnp
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    from scipy.special import expn

    from raytrax import interactions, sources, transport
    from raytrax import scene as scene_module
    from raytrax.interactions import Interactions, Kind
    from raytrax.io import vtk
    from raytrax.mesh import MeshGeometry, geometry
    from raytrax.paths import AxisymmetricPath, LinearPath
    from raytrax.scene import Scene
    from raytrax.sources import Sources

    MESHES = Path(__file__).parent / "meshes"
    FIGURES = Path(__file__).parent / "figures"

    # Mesh file, and whether it is read as the meridional half-plane.
    CASES = {
        "slab_planar": ("slab", False),
        "slab_axisymmetric": ("slab", True),
        "cylinder_coarse": ("cylinder_coarse", False),
    }

    RAY_COUNTS = (1_000, 10_000, 100_000, 1_000_000, 10_000_000)
    NUM_SEEDS = 5

    class Case(NamedTuple):
        """One case's mesh and everything a trace needs except the rays."""

        mesh: vtk.Mesh
        geom: MeshGeometry
        scene: Scene
        path: LinearPath | AxisymmetricPath
        table: Interactions
        source: Sources
        reference: float
        lanes: int | None


@app.cell(hide_code=True)
def _():
    mo.md("""
    # How many rays is enough?

    `tracing.py` shows the tangent-slab profile comes out right and `visualise.py`
    draws it; neither says how the error falls with the number of rays, nor where
    it stops falling because the mesh has become the limit. This sweeps the ray
    count over the decades either side of the 1,000,000 those notebooks use.

    Three cases, all the same physics as `visualise.py`: a hot wall at low x
    radiating into cold absorbing gas, lateral boundary mirrored so the domain is
    an infinite slab.

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
    - **binned** - RMS over the 25 axial bins `visualise.py` plots, against the
      cell-averaged `2 (E3(ka) - E3(kb)) / (b - a)`. Averaging cells first starts
      this an order of magnitude lower, but cells are sorted into bins by centroid
      and do not tile them, so the volume in a bin is not the interval the closed
      form averages over. Its centre sits off the bin midpoint, by about 2% of a
      bin on the slab and 15% on the coarse cylinder, and that misalignment is
      first order.

    First order beats second order, so the binned floor is the higher of the two
    and is reached sooner, on the slab as much as on the cylinder. At kappa = 1
    the floors are 1.0e-3 and 2.5e-3 on the slab mesh, 9.8e-3 and 2.5e-2 on the
    cylinder. Neither is a property of the rays.
    """)
    return


@app.cell
def _():
    kappa = mo.ui.slider(0.1, 5.0, 0.1, value=1.0, label="Absorptivity kappa")
    largest = mo.ui.dropdown([f"{count}" for count in RAY_COUNTS], value=f"{RAY_COUNTS[-1]}", label="Largest ray count")
    running = mo.ui.checkbox(False, label="Run the sweep")
    mo.hstack([kappa, largest, running], justify="start", gap=2)
    return kappa, largest, running


@app.function
def prepare(name):
    """Read one case and build everything a trace needs except the rays.

    `geom` is what the physics uses and is ring-weighted when revolved;
    positions stay meridional, so the profiles read `mesh.geometry` instead.
    """
    stem, axisymmetric = CASES[name]
    path = MESHES / f"{stem}.msh"
    if not path.exists():
        raise FileNotFoundError(f"{path} missing - run `uv run python examples/meshes/generate.py`")

    mesh = vtk.read(path)
    geom = geometry.revolve(mesh.geometry) if axisymmetric else mesh.geometry
    trace_path = AxisymmetricPath() if axisymmetric else LinearPath()

    scene = scene_module.build(mesh.topology, geom, mesh.groups)
    scene_module.validate(mesh.topology, geom)

    kinds = {"hot_end": Kind.ABSORB, "wall": Kind.ABSORB, "lateral": Kind.SPECULAR}
    if "axis" in mesh.groups.names:
        # Revolved, the y = 0 edge is the axis and nothing should reach it.
        # Extruded it is just another side, and has to match `lateral` or
        # the slab stops being symmetric.
        kinds["axis"] = Kind.ABSORB if axisymmetric else Kind.SPECULAR
    table = interactions.build(mesh.groups, kinds)

    hot = np.flatnonzero(mesh.groups.face_group == mesh.groups.index("hot_end"))
    source = sources.build(jnp.zeros(mesh.topology.num_cells), geom.faces.area[hot], hot)

    reference = float(source.total) / geom.cells.volume.sum()
    # Priced once per case rather than once per trace: the estimate costs two
    # compiles, and only the shape of the absorptivity reaches them.
    lanes = transport.lane_capacity(
        jax.random.key(0),
        scene,
        trace_path,
        table,
        jnp.ones(mesh.topology.num_cells),
        source,
        max_steps=2_000,
    )
    return Case(mesh, geom, scene, trace_path, table, source, reference, lanes)


@app.function
def run(case, absorptivity, rays, key):
    """Trace one case at one ray count, with the two fractions that police it.

    `max_steps` is this notebook's whole cost, not a safety net: a couple of
    mirrored rays bounce long after they carry anything, the batch pays for its
    longest, so the march runs the cap out every time. By 2,000 a full batch has
    drained to 1e-15 of the emitted energy and a refilled one has spent a
    ten-fold budget over; `spent` and `flight` are what say so on any other case.
    """
    final, deposit = transport.trace(
        key,
        case.scene,
        case.path,
        case.table,
        jnp.full(case.mesh.topology.num_cells, absorptivity),
        case.source,
        num_rays=rays,
        lanes=case.lanes,
        max_steps=2_000,
    )
    escaped = float(deposit.escaped)
    flight = float(final.energy.sum())
    spent = float(deposit.cells.sum() + deposit.faces.sum()) + escaped + flight

    heating = np.asarray(deposit.cells) / case.geom.cells.volume
    return heating, spent / float(case.source.total), flight / float(case.source.total)


@app.function
def tangent_slab(absorptivity, origin, edges):
    """Cell-averaged volumetric heating over each bin of an infinite slab."""
    near, far = absorptivity * (edges[:-1] - origin), absorptivity * (edges[1:] - origin)
    return 2 * (expn(3, near) - expn(3, far)) / (edges[1:] - edges[:-1])


@app.function
def tangent_slab_point(absorptivity, origin, position):
    """Volumetric heating at a point of an infinite slab."""
    return 2 * absorptivity * expn(2, absorptivity * (position - origin))


@app.function
def rebin(position, values, weight, edges):
    """Average `values` over the bins, weighting each cell by its measure."""
    index = np.clip(np.digitize(position, edges) - 1, 0, edges.size - 2)
    total = np.bincount(index, values * weight, edges.size - 1)
    measure = np.bincount(index, weight, edges.size - 1)
    return np.divide(total, measure, out=np.full(edges.size - 1, np.nan), where=measure > 0)


@app.function
def cell_error(case, heating, absorptivity):
    """Volume-weighted RMS error over every cell, against the point solution.

    Volume weighting is what makes the revolved reading of a mesh comparable
    with the extruded one: the cells are the same, the measures are not.
    """
    centroid, vertices = case.mesh.geometry.cells.centroid, case.mesh.geometry.vertices
    exact = tangent_slab_point(absorptivity, vertices[:, 0].min(), centroid[:, 0])
    volume = case.geom.cells.volume
    return np.sqrt(np.sum(volume * (heating - exact) ** 2) / volume.sum()) / case.reference


@app.function
def profile_error(case, heating, absorptivity, bins=26):
    """RMS error over the axial bins, against their cell-averaged solution."""
    centroid, vertices = case.mesh.geometry.cells.centroid, case.mesh.geometry.vertices
    lo, hi = vertices[:, 0].min(), vertices[:, 0].max()
    edges = np.linspace(lo, hi, bins)
    profile = rebin(centroid[:, 0], heating, case.geom.cells.volume, edges)
    return np.sqrt(np.mean((profile - tangent_slab(absorptivity, lo, edges)) ** 2)) / case.reference


@app.cell
def _(kappa, largest, running):
    def _sweep():
        if not running.value:
            return None

        counts = [count for count in RAY_COUNTS if count <= int(largest.value)]
        keys = jax.random.split(jax.random.key(0), NUM_SEEDS)
        results = {}
        with mo.status.progress_bar(total=len(CASES) * len(counts) * NUM_SEEDS) as bar:
            for name in CASES:
                case = prepare(name)
                rows = []
                for count in counts:
                    for seed, key in enumerate(keys):
                        heating, spent, flight = run(case, kappa.value, count, key)
                        rows.append(
                            (
                                count,
                                seed,
                                cell_error(case, heating, kappa.value),
                                profile_error(case, heating, kappa.value),
                                spent,
                                flight,
                            )
                        )
                        bar.update()
                results[name] = np.array(rows)
        return results

    sweep = _sweep()
    return (sweep,)


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


@app.cell(hide_code=True)
def _(sweep):
    def _table():
        if sweep is None:
            return mo.md("*Tick **Run the sweep** above. A full sweep is 75 traces, about six minutes on a GPU.*")

        header = ["rays"] + [f"{name} {metric}" for name in sweep for metric in ("cell", "binned")]
        columns = [aggregate(rows) for rows in sweep.values()]
        lines = [f"| {' | '.join(header)} |", f"|{'---|' * len(header)}"]
        for index, count in enumerate(columns[0][0]):
            cells = [f"{column[metric][index]:.2e}" for column in columns for metric in (1, 4)]
            lines.append(f"| {count:,.0f} | {' | '.join(cells)} |")

        # Unspent budget means rays that never launched; energy still in flight
        # means rays that never finished. Neither shows up in the errors as
        # anything but a floor, so they are called out rather than plotted.
        unspent = max(abs(rows[:, 4] - 1).max() for rows in sweep.values())
        flight = max(rows[:, 5].max() for rows in sweep.values())
        note = f"Worst unspent launch budget {unspent:.1e}, worst energy left in flight {flight:.1e}"
        if unspent > 1e-9 or flight > 1e-6:
            note += " - **too large**: raise `max_steps` in `run`"
        return mo.md("\n".join(lines) + f"\n\n{note}.")

    _table()
    return


@app.cell(hide_code=True)
def _(sweep):
    def _plot():
        if sweep is None:
            return mo.md("*Nothing to plot yet.*")

        fig, ax = plt.subplots(figsize=(7, 4.5))
        for index, (name, rows) in enumerate(sweep.items()):
            counts, cell, cell_lo, cell_hi, binned, binned_lo, binned_hi, _, _ = aggregate(rows)
            colour = f"C{index}"
            for values, low, high, style, label in (
                (cell, cell_lo, cell_hi, "-", f"{name}, per cell"),
                (binned, binned_lo, binned_hi, "--", f"{name}, binned"),
            ):
                ax.plot(counts, values, color=colour, ls=style, lw=2, label=label)
                ax.fill_between(counts, low, high, color=colour, alpha=0.2, lw=0)

        # One guide, on the first case, drawn last so it ends the legend. It runs
        # under the slab curves for four decades, which is the point: where a
        # curve leaves it, rays have stopped being what limits the answer.
        counts, *_, guide, _ = aggregate(next(iter(sweep.values())))
        ax.plot(counts, guide, color="0.6", lw=0.8, zorder=0, label="$N^{-1/2}$")

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Rays")
        ax.set_ylabel("RMSE / mean heating")
        ax.legend(fontsize=8)
        return fig

    _plot()
    return


@app.cell(hide_code=True)
def _():
    write = mo.ui.run_button(label="Write every CSV")
    mo.md(f"""
    ## Export

    One folder per case under `examples/figures/convergence/`, each with
    `rmse.csv` (one row per ray count, seeds collapsed to mean and range, plus
    the `N^-1/2` guide as its own column) and `seeds.csv` (the raw runs). Every
    error is already divided by the mean heating, so the columns are
    dimensionless.

    Nothing is rendered: the convention is that line data leaves as CSV and is
    replotted in Typst, and only field renders leave as images.

    {write}
    """)
    return (write,)


@app.cell(hide_code=True)
def _(sweep, write):
    def _save(path, columns, header):
        """Write one flat CSV of the named columns."""
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savetxt(path, np.column_stack(columns), delimiter=",", header=header, comments="", fmt="%.8g")
        return path

    def _export():
        if not write.value:
            return mo.md("*Not written yet.*")
        if sweep is None:
            return mo.md("*Run the sweep first.*")

        written = []
        for name, rows in sweep.items():
            case_dir = FIGURES / "convergence" / name
            written.append(
                _save(
                    case_dir / "rmse.csv",
                    aggregate(rows),
                    "rays,cell_rmse,cell_lo,cell_hi,bin_rmse,bin_lo,bin_hi,cell_sqrt_law,bin_sqrt_law",
                )
            )
            written.append(_save(case_dir / "seeds.csv", rows.T, "rays,seed,cell_rmse,bin_rmse,spent,in_flight"))
        return mo.md("Wrote:\n\n" + "\n".join(f"- `{path.relative_to(FIGURES)}`" for path in written))

    _export()
    return


if __name__ == "__main__":
    app.run()
