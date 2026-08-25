import marimo

__generated_with = "0.24.0"
app = marimo.App(width="medium")

with app.setup:
    import itertools
    import os
    from pathlib import Path

    import jax
    import jax.numpy as jnp
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    from scipy.special import expn

    # Must be set before pyvista imports vtk, and before any Plotter is built.
    os.environ.setdefault("PYVISTA_OFF_SCREEN", "true")
    import pyvista as pv

    from raytrax import interactions, sources, transport
    from raytrax import scene as scene_module
    from raytrax.interactions import Kind
    from raytrax.io import vtk
    from raytrax.mesh import geometry
    from raytrax.paths import RADIAL, AxisymmetricPath, LinearPath

    # Never negotiate a jupyter backend: marimo does not render ipywidgets, so
    # every figure here is a screenshot taken off screen.
    pv.OFF_SCREEN = True
    pv.global_theme.notebook = False

    MESHES = Path(__file__).parent / "meshes"
    FIGURES = Path(__file__).parent / "figures"

    # Mesh file, and whether it is read as the meridional half-plane.
    CASES = {
        "slab_planar": ("slab", False),
        "slab_axisymmetric": ("slab", True),
        "cylinder_coarse": ("cylinder_coarse", False),
        "cylinder": ("cylinder", False),
    }

    VTK_CELL = {
        "triangle": pv.CellType.TRIANGLE,
        "quad": pv.CellType.QUAD,
        "tetra": pv.CellType.TETRA,
        "hexahedron": pv.CellType.HEXAHEDRON,
        "wedge": pv.CellType.WEDGE,
        "pyramid": pv.CellType.PYRAMID,
    }


@app.cell(hide_code=True)
def _():
    mo.md("""
    # Figures for the tangent-slab cases

    `tracing.py` proves the physics. This notebook turns the same problem into
    slide material: pyvista renders written straight to PNG, and the profiles
    written to CSV so they can be replotted anywhere.

    - **slab_planar** and **slab_axisymmetric** - one jumbled 2D mesh, triangles
      scattered among quads at varying cell size, read two different ways. Both
      are drawn flat, the axisymmetric one as the meridional `(x, r)`
      half-plane it is solved in.
    - **cylinder_coarse** - 3D tets, cut in half so whole cells survive on the
      cut face. Coarse, because the point is the cells.
    - **cylinder** - the same at full resolution, for the heating field.

    Every case is the same problem: a hot wall at low x radiating into cold
    absorbing gas, with the lateral boundary mirrored so the domain is an
    infinite slab and the profile has a closed form.

    Figures are written with a transparent background, so they drop onto any
    slide. The scalar bar and legend text stay black.
    """)
    return


@app.cell
def _():
    case = mo.ui.dropdown(list(CASES), value="slab_planar", label="Case")
    kappa = mo.ui.slider(0.1, 5.0, 0.1, value=1.0, label="Absorptivity kappa")
    num_rays = mo.ui.dropdown(["50000", "200000", "1000000"], value="200000", label="Rays")
    ragged = mo.ui.checkbox(True, label="Ragged cut (whole cells)")
    mo.hstack([case, kappa, num_rays, ragged], justify="start", gap=2)
    return case, kappa, num_rays, ragged


@app.cell
def _():
    def solve(name, absorptivity, rays):
        """Trace one case, returning the mesh, its weighting, and cell heating.

        `geom` is what the physics uses and is ring-weighted when revolved;
        positions stay meridional, so plots read `mesh.geometry` instead.
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

        _, deposit = transport.trace(
            jax.random.key(4),
            scene,
            trace_path,
            table,
            jnp.full(mesh.topology.num_cells, absorptivity),
            source,
            num_rays=rays,
            max_steps=4000,
        )
        return mesh, geom, np.asarray(deposit.cells) / geom.cells.volume

    return (solve,)


@app.cell
def _(case, kappa, num_rays, solve):
    mesh, geom, heating = solve(case.value, kappa.value, int(num_rays.value))
    axisymmetric = CASES[case.value][1]
    return axisymmetric, geom, heating, mesh


@app.cell(hide_code=True)
def _(geom, heating, mesh):
    def _stats():
        counts: dict[str, int] = {}
        for index, element in enumerate(mesh.topology.block_elements):
            counts[element.name] = counts.get(element.name, 0) + int((mesh.topology.cell_block == index).sum())
        measure = geom.cells.volume
        return mo.md(f"""
        | | |
        |---|---|
        | cells | {mesh.topology.num_cells:,} - {", ".join(f"{n:,} {t}" for t, n in counts.items())} |
        | boundary groups | {", ".join(mesh.groups.names[1:])} |
        | cell size spread | {measure.max() / measure.min():.1f}x |
        | peak heating | {heating.max():.4f} |
        """)

    _stats()
    return


@app.cell
def _():
    def to_pyvista(mesh, values=None):
        """Wrap a raytrax mesh as a pyvista grid, cell for cell in the same order."""
        topology = mesh.topology
        cells, types = [], []
        for index, element in enumerate(topology.block_elements):
            rows = np.flatnonzero(topology.cell_block == index)
            connectivity = topology.cell_vertices[rows, : element.num_vertices]
            counts = np.full((len(rows), 1), element.num_vertices)
            cells.append(np.hstack([counts, connectivity]).ravel())
            types.append(np.full(len(rows), VTK_CELL[element.name], dtype=np.uint8))

        # A 2D mesh stores two columns; VTK always wants three.
        points = np.zeros((len(mesh.geometry.vertices), 3))
        points[:, : topology.dim] = mesh.geometry.vertices

        grid = pv.UnstructuredGrid(np.concatenate(cells), np.concatenate(types), points)
        if values is not None:
            grid.cell_data["heating"] = values
        return grid

    def half_cut(grid, ragged=True):
        """Keep the y <= 0 half, either by whole cells or by slicing through them."""
        if not ragged:
            return grid.clip(normal="y", origin=(0.0, 0.0, 0.0))
        # Any vertex at y <= 0 keeps the whole cell, so the cut face is the cell
        # boundary itself: stair-stepped, and it bulges past y = 0.
        return grid.extract_points(grid.points[:, 1] <= 0.0, adjacent_cells=True)

    def render(build, path=None, size=(1400, 900), scale=2, transparent=True):
        """Run `build(plotter)` off screen, returning the frame and optionally saving it.

        Transparent by default so a figure drops onto any slide background. The
        scalar bar and legend stay black, so a dark slide needs `transparent`
        off, or its own text colour.
        """
        plotter = pv.Plotter(off_screen=True, window_size=list(size))
        try:
            build(plotter)
            return plotter.screenshot(
                None if path is None else str(path),
                return_img=True,
                scale=scale,
                transparent_background=transparent,
            )
        finally:
            # marimo re-runs a cell on any upstream change, and a Plotter left
            # open holds its render window.
            plotter.close()

    return half_cut, render, to_pyvista


@app.cell
def _(half_cut, to_pyvista):
    def bar_args(label):
        """Scalar bar in the right-hand margin, clear of the geometry."""
        return {
            "title": label,
            "vertical": True,
            "color": "black",
            "title_font_size": 20,
            "label_font_size": 16,
            "position_x": 0.87,
            "position_y": 0.25,
            "width": 0.04,
            "height": 0.5,
        }

    def frame(plotter, three_d):
        """Fit the view, leaving the margin the scalar bar is placed in.

        `camera.tight` would fill the frame edge to edge, which puts the bar on
        top of the geometry, and in 3D it discards the camera set here.
        """
        if three_d:
            # The cut face points along +y, so the camera sits on that side to
            # see it, tilted towards the hot end so the gradient runs across.
            plotter.camera_position = [(-0.9, 2.5, 1.0), (0.0, 0.0, 0.0), (0.0, 0.0, 1.0)]
        else:
            plotter.view_xy()
        plotter.reset_camera()
        plotter.camera.zoom(0.92)

    def draw(plotter, mesh, values=None, ragged=True, cmap="magma", label="Volumetric heating", bar=True):
        """Add one case to a plotter: cut in half if it is 3D, face on if it is 2D."""
        grid = to_pyvista(mesh, values)
        if values is None:
            colours = {"color": "#d9d9d9"}
        else:
            colours = {"scalars": "heating", "cmap": cmap, "show_scalar_bar": bar}
            if bar:
                colours["scalar_bar_args"] = bar_args(label)

        # Before any camera work: the parallel scale is set when projection is
        # switched, so switching afterwards throws the framing away.
        plotter.enable_parallel_projection()
        plotter.set_background("white")

        if mesh.topology.dim == 3:
            piece = half_cut(grid, ragged=ragged)
            assert values is None or "heating" in piece.cell_data, "cell field lost in extraction"
            plotter.add_mesh(piece, show_edges=True, edge_color="#262626", line_width=0.4, **colours)
            # A faint skin of the whole cylinder, so the cut reads as a cut.
            plotter.add_mesh(
                grid.extract_surface(algorithm="dataset_surface"),
                style="wireframe",
                color="#b0b0b0",
                line_width=0.4,
                opacity=0.2,
            )
        else:
            # Flat colour: with lighting on, VTK shades each polygon and the
            # colour stops reading off the colour bar.
            plotter.add_mesh(grid, show_edges=True, edge_color="#262626", line_width=1.0, lighting=False, **colours)

        frame(plotter, mesh.topology.dim == 3)

    def element_vertices(mesh):
        """Vertices per cell, so a mixed mesh can be coloured by element type."""
        counts = np.zeros(mesh.topology.num_cells, dtype=float)
        for index, element in enumerate(mesh.topology.block_elements):
            counts[mesh.topology.cell_block == index] = element.num_vertices
        return counts

    def draw_mesh(plotter, mesh, ragged=True):
        """The bare mesh, coloured by element type where there is more than one."""
        counts = element_vertices(mesh)
        names = {3: "triangle", 4: "quad", 5: "pyramid", 6: "wedge", 8: "hexahedron"}
        present = sorted(np.unique(counts).astype(int))
        if len(present) < 2:
            return draw(plotter, mesh, ragged=ragged)

        # Two element types, so two flat colours and a legend, not a ramp.
        palette = ["#4c72b0", "#dd8452", "#55a868", "#c44e52"]
        draw(plotter, mesh, counts, ragged, cmap=palette[: len(present)], label="", bar=False)
        plotter.add_legend(
            [[names.get(count, str(count)), colour] for count, colour in zip(present, palette, strict=False)],
            bcolor="white",
            border=True,
            size=(0.13, 0.09),
            loc="upper right",
        )

    return draw, draw_mesh


@app.cell(hide_code=True)
def _(draw_mesh, mesh, ragged, render):
    mo.image(render(lambda plotter: draw_mesh(plotter, mesh, ragged.value)), width=900)
    return


@app.cell(hide_code=True)
def _(draw, heating, mesh, ragged, render):
    mo.image(render(lambda plotter: draw(plotter, mesh, heating, ragged.value)), width=900)
    return


@app.cell
def _():
    def tangent_slab(absorptivity, origin, edges):
        """Cell-averaged volumetric heating over each bin of an infinite slab."""
        near, far = absorptivity * (edges[:-1] - origin), absorptivity * (edges[1:] - origin)
        return 2 * (expn(3, near) - expn(3, far)) / (edges[1:] - edges[:-1])

    def rebin(position, values, weight, edges):
        """Average `values` over the bins, weighting each cell by its measure."""
        index = np.clip(np.digitize(position, edges) - 1, 0, edges.size - 2)
        total = np.bincount(index, values * weight, edges.size - 1)
        measure = np.bincount(index, weight, edges.size - 1)
        return np.divide(total, measure, out=np.full(edges.size - 1, np.nan), where=measure > 0)

    def save(path, columns, header):
        """Write one flat CSV of the named columns."""
        FIGURES.mkdir(exist_ok=True)
        np.savetxt(path, np.column_stack(columns), delimiter=",", header=header, comments="", fmt="%.8g")
        return path

    def axial_csv(path, edges, profile, exact):
        """One row per axial bin."""
        mid = (edges[:-1] + edges[1:]) / 2
        return save(path, (edges[:-1], edges[1:], mid, profile, exact), "lo,hi,mid,heating,tangent_slab")

    def radial_csv(path, shells, rows):
        """One row per shell, repeated for each axial band."""
        mid = (shells[:-1] + shells[1:]) / 2
        bands = [
            np.column_stack(
                [
                    np.full(mid.size, lo),
                    np.full(mid.size, hi),
                    shells[:-1],
                    shells[1:],
                    mid,
                    shell,
                    np.full(mid.size, flat),
                ]
            )
            for lo, hi, shell, flat in rows
        ]
        return save(path, np.vstack(bands).T, "band_lo,band_hi,lo,hi,mid,heating,tangent_slab")

    return axial_csv, radial_csv, rebin, tangent_slab


@app.cell
def _(rebin, tangent_slab):
    def axial_profile(mesh, geom, heating, absorptivity, bins=26):
        """Monte-Carlo and analytic heating against depth, bin for bin."""
        centroid, vertices = mesh.geometry.cells.centroid, mesh.geometry.vertices
        lo, hi = vertices[:, 0].min(), vertices[:, 0].max()
        edges = np.linspace(lo, hi, bins)
        profile = rebin(centroid[:, 0], heating, geom.cells.volume, edges)
        return edges, profile, tangent_slab(absorptivity, lo, edges)

    def radial_profile(mesh, geom, heating, absorptivity, axisymmetric, bands=5):
        """Heating against radius in a few axial bands. None unless there is an axis."""
        if mesh.topology.dim != 3 and not axisymmetric:
            return None

        centroid, vertices = mesh.geometry.cells.centroid, mesh.geometry.vertices
        # Revolved, the radius is stored outright; in 3D it has to be measured.
        if axisymmetric:
            radius, edge = centroid[:, RADIAL], vertices[:, RADIAL]
        else:
            radius, edge = np.linalg.norm(centroid[:, 1:], axis=-1), np.linalg.norm(vertices[:, 1:], axis=-1)

        lo = vertices[:, 0].min()
        band_edges = np.linspace(lo, vertices[:, 0].max(), bands)
        shells = np.linspace(0.0, edge.max(), 31 if mesh.topology.dim == 3 else 13)

        rows = []
        for a, b in itertools.pairwise(band_edges):
            inside = (centroid[:, 0] >= a) & (centroid[:, 0] < b)
            profile = rebin(radius[inside], heating[inside], geom.cells.volume[inside], shells)
            # The slab solution is radially uniform, so it is one flat value.
            rows.append((a, b, profile, tangent_slab(absorptivity, lo, np.array([a, b]))[0]))
        return shells, rows

    return axial_profile, radial_profile


@app.cell(hide_code=True)
def _(axial_profile, geom, heating, kappa, mesh):
    def _axial():
        edges, profile, exact = axial_profile(mesh, geom, heating, kappa.value)
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.stairs(profile, edges, label="Monte-Carlo", lw=2)
        ax.stairs(exact, edges, label="Tangent slab", ls="--", color="C1", lw=2)
        ax.set_xlabel("x")
        ax.set_ylabel("Volumetric heating")
        ax.legend()
        return fig

    _axial()
    return


@app.cell(hide_code=True)
def _(axisymmetric, geom, heating, kappa, mesh, radial_profile):
    def _radial():
        result = radial_profile(mesh, geom, heating, kappa.value, axisymmetric)
        if result is None:
            return mo.md("*Radial shells need an axis: the cylinder or the axisymmetric case.*")

        shells, rows = result
        fig, ax = plt.subplots(figsize=(7, 4))
        for i, (a, b, profile, flat) in enumerate(rows):
            ax.stairs(profile, shells, color=f"C{i}", label=f"${a:+.2f} < x < {b:+.2f}$")
            ax.axhline(flat, color=f"C{i}", ls="--", lw=0.8)
        ax.set_xlabel("r")
        ax.set_ylabel("Volumetric heating")
        ax.legend(fontsize=8)
        return fig

    _radial()
    return


@app.cell(hide_code=True)
def _():
    write = mo.ui.run_button(label="Write every figure and CSV")
    mo.md(f"""
    ## Export

    Renders all four cases at the current kappa and ray count into
    `examples/figures/`. This retraces each case, so it takes a minute.

    {write}
    """)
    return (write,)


@app.cell(hide_code=True)
def _(
    axial_csv,
    axial_profile,
    draw,
    draw_mesh,
    kappa,
    num_rays,
    radial_csv,
    radial_profile,
    ragged,
    render,
    solve,
    write,
):
    def _shoot(name, suffix, build):
        path = FIGURES / f"{name}_{suffix}.png"
        render(build, path, size=(1600, 1000))
        return path

    def _export_case(name, axisymmetric):
        mesh, geom, heating = solve(name, kappa.value, int(num_rays.value))
        written = [
            _shoot(name, "mesh", lambda plotter: draw_mesh(plotter, mesh, ragged.value)),
            _shoot(name, "heating", lambda plotter: draw(plotter, mesh, heating, ragged.value)),
        ]
        written.append(axial_csv(FIGURES / f"axial_{name}.csv", *axial_profile(mesh, geom, heating, kappa.value)))
        radial = radial_profile(mesh, geom, heating, kappa.value, axisymmetric)
        if radial is not None:
            written.append(radial_csv(FIGURES / f"radial_{name}.csv", *radial))
        return written

    def _export():
        if not write.value:
            return mo.md("*Not written yet.*")
        FIGURES.mkdir(exist_ok=True)
        written = [path for name, (_, axi) in CASES.items() for path in _export_case(name, axi)]
        return mo.md("Wrote:\n\n" + "\n".join(f"- `{path.name}`" for path in written))

    _export()
    return


if __name__ == "__main__":
    app.run()
