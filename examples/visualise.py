import marimo

__generated_with = "0.24.0"
app = marimo.App(width="medium")

with app.setup:
    import itertools
    import json
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

    # The cut face points along +y, so the camera sits on that side to see it,
    # tilted towards the hot end so the gradient runs across. Position, focal
    # point, view up -- pyvista's camera_position format.
    CYLINDER_CAMERA = ((-0.9, 2.5, 1.0), (0.0, 0.0, 0.0), (0.0, 0.0, 1.0))


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


@app.cell(hide_code=True)
def _():
    mo.md("""
    ## Why these meshes

    `slab` is the mixed-element case worth putting on a slide. Triangles are
    scattered among quads rather than split off into their own half, the cell
    size varies across the domain, and it is conformal throughout: every
    interior face has exactly two cells on it, whatever their element types.
    One file, and it reads either as an extruded slab or, revolved about
    `y = 0`, as the meridional half-plane of the cylinder. Those are the two
    `slab_*` cases, and they are the same 1,593 cells either way.

    `cylinder_coarse` exists only to be looked at. The full `cylinder` is
    34,507 tets, and a ragged half-cut of that many is a grey smear at slide
    size; at 2,711 the individual cells survive on the cut face. The heating
    field is taken from the full mesh instead, where the resolution is the
    point and the cells are not.

    ## Running this notebook

    Unlike `tracing.py`, this one needs the optional pyvista stack:

    ```
    uv sync --extra visualisation
    ```

    Every figure here is a screenshot taken off screen. marimo does not render
    ipywidgets, so there is no interactive VTK backend to fall back on, and
    `PYVISTA_OFF_SCREEN` is set in the setup block above before pyvista ever
    imports vtk. Off screen still means a real GL context, so this wants a
    display: a headless machine needs `xvfb-run`, or the `vtk-osmesa` wheel in
    place of `vtk`.
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


@app.function
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


@app.cell
def _(case, kappa, num_rays):
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

    def mesh_extent(mesh, margin=0.04):
        """Padded (xmin, ymin, xmax, ymax) of the mesh's own vertices, x and y only.

        Read off the vertices rather than a camera, so it does not depend on
        framing: slab_planar and slab_axisymmetric are the same file, and
        always come out with the same box.
        """
        vertices = mesh.geometry.vertices[:, :2]
        (xmin, ymin), (xmax, ymax) = vertices.min(axis=0), vertices.max(axis=0)
        pad_x, pad_y = (xmax - xmin) * margin, (ymax - ymin) * margin
        return xmin - pad_x, ymin - pad_y, xmax + pad_x, ymax + pad_y

    def fit_window(bounds, pixels=1600):
        """Window size in pixels matching the aspect ratio of `bounds`.

        A parallel camera's horizontal extent is its vertical extent times the
        window's aspect ratio, so an exact tight fit needs the window shaped to
        match the bounds before the camera is ever set.
        """
        xmin, ymin, xmax, ymax = bounds
        width, height = xmax - xmin, ymax - ymin
        scale = pixels / max(width, height)
        return max(round(width * scale), 1), max(round(height * scale), 1)

    def projected_extent(mesh, camera_position, margin=0.04):
        """The mesh's vertices, projected onto the camera's (right, up) view plane.

        A tilted camera's visible frame is not axis-aligned in `(x, y, z)`, so
        fitting a window to the 3D bounding box leaves whatever margin the
        axis the camera looks along happens to contribute. Projecting onto the
        view plane first gives the actual 2D footprint that needs to fit,
        which `fit_window` can then match exactly, same as the 2D cases.
        """
        position, focal, up = (np.array(v, dtype=float) for v in camera_position)
        forward = focal - position
        forward /= np.linalg.norm(forward)
        right = np.cross(forward, up)
        right /= np.linalg.norm(right)
        up = np.cross(right, forward)

        u = mesh.geometry.vertices @ right
        v = mesh.geometry.vertices @ up
        pad_u, pad_v = (u.max() - u.min()) * margin, (v.max() - v.min()) * margin
        return u.min() - pad_u, v.min() - pad_v, u.max() + pad_u, v.max() + pad_v

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

    return (
        fit_window,
        half_cut,
        mesh_extent,
        projected_extent,
        render,
        to_pyvista,
    )


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

    def frame(plotter, three_d, tight_bounds=None):
        """Fit the view, leaving the margin the scalar bar is placed in.

        `camera.tight` would fill the frame edge to edge, which puts the bar on
        top of the geometry, and in 3D it discards the camera set here.
        `tight_bounds` fits the camera exactly to the mesh instead, for bare
        exports where nothing sits in that margin: face on, it is a
        world-space `(xmin, ymin, xmax, ymax)`; tilted, it is the mesh's own
        footprint on the camera's view plane, from `projected_extent`.

        Tilted still calls `reset_camera()` first, for its correct centring --
        but its `parallel_scale` fits the bounding box's *diagonal*, which is
        looser than the actual footprint for any shape that is not roughly
        cubic (a cylinder cut in half is not), so that gets overwritten with
        the exact value once the window is sized to hold it tight.
        """
        if tight_bounds is not None and not three_d:
            xmin, ymin, xmax, ymax = tight_bounds
            cx, cy = (xmin + xmax) / 2, (ymin + ymax) / 2
            plotter.camera_position = [(cx, cy, 1.0), (cx, cy, 0.0), (0.0, 1.0, 0.0)]
            plotter.camera.parallel_scale = (ymax - ymin) / 2
            return
        if three_d:
            plotter.camera_position = list(CYLINDER_CAMERA)
        else:
            plotter.view_xy()
        plotter.reset_camera()
        if tight_bounds is not None:
            _, vmin, _, vmax = tight_bounds
            plotter.camera.parallel_scale = (vmax - vmin) / 2
        else:
            plotter.camera.zoom(0.92)

    def draw(
        plotter,
        mesh,
        values=None,
        ragged=True,
        cmap="magma",
        label="Volumetric heating",
        bar=True,
        tight_bounds=None,
        symmetric=False,
    ):
        """Add one case to a plotter: cut in half if it is 3D, face on if it is 2D.

        `symmetric` centres the colour range on zero at `+/- max(abs(values))`,
        for a signed field (a difference) where zero should sit on a fixed
        colour rather than wherever the data happens to average out.
        """
        grid = to_pyvista(mesh, values)
        if values is None:
            colours = {"color": "#d9d9d9"}
        else:
            # Pinned to the full field, not the piece pyvista actually draws:
            # `half_cut` in 3D would otherwise narrow the range to whatever
            # survives the cut, and the exported colorbar range would lie.
            if symmetric:
                bound = float(np.nanmax(np.abs(values)))
                clim = (-bound, bound)
            else:
                clim = (float(np.nanmin(values)), float(np.nanmax(values)))
            colours = {"scalars": "heating", "cmap": cmap, "show_scalar_bar": bar, "clim": clim}
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

        frame(plotter, mesh.topology.dim == 3, tight_bounds=tight_bounds)

    def element_vertices(mesh):
        """Vertices per cell, so a mixed mesh can be coloured by element type."""
        counts = np.zeros(mesh.topology.num_cells, dtype=float)
        for index, element in enumerate(mesh.topology.block_elements):
            counts[mesh.topology.cell_block == index] = element.num_vertices
        return counts

    def draw_mesh(plotter, mesh, ragged=True, legend=True, tight_bounds=None):
        """The bare mesh, coloured by element type where there is more than one.

        Returns the legend entries (label and colour, one per element type), or
        `None` where there is only one type to draw. `legend=False` skips
        drawing the on-image legend actor while still returning its contents,
        for callers that want the data without the picture.
        """
        counts = element_vertices(mesh)
        names = {3: "triangle", 4: "quad", 5: "pyramid", 6: "wedge", 8: "hexahedron"}
        present = sorted(np.unique(counts).astype(int))
        if len(present) < 2:
            draw(plotter, mesh, ragged=ragged, tight_bounds=tight_bounds)
            return None

        # Two element types, so two flat colours and a legend, not a ramp.
        palette = ["#4c72b0", "#dd8452", "#55a868", "#c44e52"]
        draw(
            plotter, mesh, counts, ragged, cmap=palette[: len(present)], label="", bar=False, tight_bounds=tight_bounds
        )
        entries = [
            {"label": names.get(count, str(count)), "colour": colour}
            for count, colour in zip(present, palette, strict=False)
        ]
        if legend:
            plotter.add_legend(
                [[entry["label"], entry["colour"]] for entry in entries],
                bcolor="white",
                border=True,
                size=(0.13, 0.09),
                loc="upper right",
            )
        return entries

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
        path.parent.mkdir(parents=True, exist_ok=True)
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

    return axial_csv, radial_csv, rebin, save, tangent_slab


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
    mo.md("""
    ## Planar against axisymmetric

    `slab_planar` and `slab_axisymmetric` are the same 1,593 cells, the same
    trace, the same everything except how the geometry is read: extruded flat
    for one, revolved about `y = 0` for the other. Cell for cell, in the same
    order, so the two heating fields subtract directly with no interpolation.

    Any difference is the axisymmetric geometry, not sampling noise: revolving
    the mesh turns each `y` into a radius, so a cell's volume is its flat area
    times the ring it sweeps, vanishing towards `y = 0` rather than staying
    uniform the way the planar reading treats it. Heating divides deposited
    energy by that volume, so cells nearest the axis are where the two
    readings are most exposed to disagreeing -- watch that edge in the figure
    below, not the wall end of the domain.
    """)
    return


@app.cell
def _(kappa, num_rays):
    planar_mesh, _planar_geom, planar_heating = solve("slab_planar", kappa.value, int(num_rays.value))
    _axisym_mesh, _axisym_geom, axisym_heating = solve("slab_axisymmetric", kappa.value, int(num_rays.value))
    return axisym_heating, planar_heating, planar_mesh


@app.cell
def _(axisym_heating, planar_heating):
    def compare(planar, axisymmetric):
        """Signed difference, and that difference as a fraction of the peak planar value.

        Normalised against the peak rather than the local value: heating decays
        by orders of magnitude away from the wall, so dividing cell by cell
        would blow up wherever the planar reading is near zero, without saying
        anything new about the size of the disagreement.
        """
        diff = axisymmetric - planar
        peak = float(np.max(np.abs(planar)))
        return diff, diff / peak

    diff, relative = compare(planar_heating, axisym_heating)
    return diff, relative


@app.cell(hide_code=True)
def _(diff, planar_heating, relative):
    mo.md(f"""
    | | |
    |---|---|
    | max absolute difference | {np.max(np.abs(diff)):.4f} 1/m |
    | mean absolute difference | {np.mean(np.abs(diff)):.4f} 1/m |
    | max relative to peak planar heating | {np.max(np.abs(relative)):.2%} |
    | peak planar heating | {planar_heating.max():.4f} 1/m |
    """)
    return


@app.cell(hide_code=True)
def _(diff, draw, planar_mesh, ragged, relative, render):
    mo.hstack(
        [
            mo.image(
                render(
                    lambda plotter: draw(
                        plotter,
                        planar_mesh,
                        diff,
                        ragged.value,
                        cmap="RdBu_r",
                        label="Difference (1/m)",
                        symmetric=True,
                    )
                ),
                width=440,
            ),
            mo.image(
                render(
                    lambda plotter: draw(
                        plotter,
                        planar_mesh,
                        relative,
                        ragged.value,
                        cmap="RdBu_r",
                        label="Relative to peak planar",
                        symmetric=True,
                    )
                ),
                width=440,
            ),
        ],
        justify="start",
        gap=2,
    )
    return


@app.cell(hide_code=True)
def _():
    write = mo.ui.run_button(label="Write every figure and CSV")
    mo.md(f"""
    ## Export

    Renders all four cases at the current kappa and ray count, one folder per
    case under `examples/figures/` (`slab_planar/`, `slab_axisymmetric/`,
    `cylinder_coarse/`, `cylinder/`), each with `mesh.png`, `heating.png`,
    `axial.csv`, and, where there is an axis, `radial.csv`. This retraces each
    case, so it takes a minute.

    The mesh and heating renders hold only the geometry -- no scalar bar or
    legend -- so each comes with a `.json` sidecar giving the legend contents
    or the colorbar range. The two `slab_*` cases are shot face on, and also
    get the `(xmin, ymin, xmax, ymax)` world bounds of that view for
    alignment; the cylinder cases use a tilted camera, so there is no flat
    viewbox to give.

    A fifth folder, `slab_comparison/`, holds the planar-against-axisymmetric
    comparison above: `difference.png` and `relative.png`, each bare with a
    `.json` sidecar, plus `comparison.csv` with both fields and their
    difference, one row per cell.

    {write}
    """)
    return (write,)


@app.cell(hide_code=True)
def _(
    axial_csv,
    axial_profile,
    axisym_heating,
    diff,
    draw,
    draw_mesh,
    fit_window,
    kappa,
    mesh_extent,
    num_rays,
    planar_heating,
    planar_mesh,
    projected_extent,
    radial_csv,
    radial_profile,
    ragged,
    relative,
    render,
    save,
    write,
):
    def _save_json(path, data):
        path.write_text(json.dumps(data, indent=2))
        return path

    def _colorbar(heating):
        # `solve`'s hot_end source is face area with sigma T^4 = 1, not a real
        # power, so deposit carries units of face measure rather than W. Heating
        # is deposit over cell volume, one dimension short of W/m^3 either way
        # (length over area in 2D, area over volume in 3D), so it comes out 1/m.
        return {
            "min": float(heating.min()),
            "max": float(heating.max()),
            "cmap": "magma",
            "label": "Volumetric heating (1/m)",
        }

    def _export_2d(case_dir, mesh, geom, heating, axisymmetric):
        """Face-on cases: an exact camera fit, so the bounds are worth writing out."""
        bounds = mesh_extent(mesh)
        xmin, ymin, xmax, ymax = bounds
        size = fit_window(bounds)
        written = []

        entries = None

        def _mesh_build(plotter):
            nonlocal entries
            entries = draw_mesh(plotter, mesh, ragged.value, legend=False, tight_bounds=bounds)

        mesh_path = case_dir / "mesh.png"
        render(_mesh_build, mesh_path, size=size)
        written.append(mesh_path)
        mesh_meta = {"xmin": xmin, "ymin": ymin, "xmax": xmax, "ymax": ymax}
        if entries:
            mesh_meta["legend"] = entries
        written.append(_save_json(case_dir / "mesh.json", mesh_meta))

        heating_path = case_dir / "heating.png"
        render(lambda plotter: draw(plotter, mesh, heating, bar=False, tight_bounds=bounds), heating_path, size=size)
        written.append(heating_path)
        heating_meta = {"xmin": xmin, "ymin": ymin, "xmax": xmax, "ymax": ymax, "colorbar": _colorbar(heating)}
        written.append(_save_json(case_dir / "heating.json", heating_meta))

        written.append(axial_csv(case_dir / "axial.csv", *axial_profile(mesh, geom, heating, kappa.value)))
        radial = radial_profile(mesh, geom, heating, kappa.value, axisymmetric)
        if radial is not None:
            written.append(radial_csv(case_dir / "radial.csv", *radial))
        return written

    def _export_3d(case_dir, mesh, geom, heating):
        """Tilted-camera cases: no flat viewbox to give, so no bounds JSON.

        The window is sized to the mesh's own footprint (projected onto the
        tilted camera's view plane, not the raw 3D bounds), and that same
        footprint is passed on as `tight_bounds` so `frame` can overwrite
        `reset_camera`'s loose diagonal-fit scale with the exact one -- without
        both, a cut-in-half cylinder still renders into a small fraction of
        the frame no matter how the window is shaped.
        """
        written = []
        extent = projected_extent(mesh, CYLINDER_CAMERA)
        size = fit_window(extent)

        mesh_path = case_dir / "mesh.png"
        render(
            lambda plotter: draw_mesh(plotter, mesh, ragged.value, legend=False, tight_bounds=extent),
            mesh_path,
            size=size,
        )
        written.append(mesh_path)
        # cylinder cases are tets only, so draw_mesh never has a legend to report

        heating_path = case_dir / "heating.png"
        render(
            lambda plotter: draw(plotter, mesh, heating, ragged.value, bar=False, tight_bounds=extent),
            heating_path,
            size=size,
        )
        written.append(heating_path)
        written.append(_save_json(case_dir / "heating.json", {"colorbar": _colorbar(heating)}))

        written.append(axial_csv(case_dir / "axial.csv", *axial_profile(mesh, geom, heating, kappa.value)))
        radial = radial_profile(mesh, geom, heating, kappa.value, axisymmetric=False)
        if radial is not None:
            written.append(radial_csv(case_dir / "radial.csv", *radial))
        return written

    def _export_case(name, axisymmetric):
        case_dir = FIGURES / name
        case_dir.mkdir(parents=True, exist_ok=True)
        mesh, geom, heating = solve(name, kappa.value, int(num_rays.value))
        if mesh.topology.dim == 3:
            return _export_3d(case_dir, mesh, geom, heating)
        return _export_2d(case_dir, mesh, geom, heating, axisymmetric)

    def _export_field(case_dir, stem, mesh, values, bounds, label):
        """One signed field, bare and centred on zero, with its own colorbar range."""
        size = fit_window(bounds)
        path = case_dir / f"{stem}.png"
        render(
            lambda plotter: draw(plotter, mesh, values, cmap="RdBu_r", bar=False, tight_bounds=bounds, symmetric=True),
            path,
            size=size,
        )
        bound = float(np.max(np.abs(values)))
        xmin, ymin, xmax, ymax = bounds
        meta = {
            "xmin": xmin,
            "ymin": ymin,
            "xmax": xmax,
            "ymax": ymax,
            "colorbar": {"min": -bound, "max": bound, "cmap": "RdBu_r", "label": label},
        }
        return [path, _save_json(case_dir / f"{stem}.json", meta)]

    def _export_comparison():
        """Planar against axisymmetric: same cells, same order, so a direct subtraction."""
        case_dir = FIGURES / "slab_comparison"
        case_dir.mkdir(parents=True, exist_ok=True)
        bounds = mesh_extent(planar_mesh)

        written = _export_field(case_dir, "difference", planar_mesh, diff, bounds, "Difference (1/m)")
        written += _export_field(case_dir, "relative", planar_mesh, relative, bounds, "Relative to peak planar heating")

        centroid = planar_mesh.geometry.cells.centroid
        written.append(
            save(
                case_dir / "comparison.csv",
                (centroid[:, 0], centroid[:, 1], planar_heating, axisym_heating, diff, relative),
                "x,y,planar,axisymmetric,diff,relative",
            )
        )
        return written

    def _export():
        if not write.value:
            return mo.md("*Not written yet.*")
        written = [path for name, (_, axi) in CASES.items() for path in _export_case(name, axi)]
        written += _export_comparison()
        return mo.md("Wrote:\n\n" + "\n".join(f"- `{path.relative_to(FIGURES)}`" for path in written))

    _export()
    return


if __name__ == "__main__":
    app.run()
