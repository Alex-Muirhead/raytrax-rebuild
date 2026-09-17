import marimo

__generated_with = "0.24.0"
app = marimo.App(width="medium")

with app.setup:
    import json
    from pathlib import Path

    import jax
    import jax.numpy as jnp
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.collections import PolyCollection

    from raytrax import interactions, sources, transport
    from raytrax import scene as scene_module
    from raytrax.interactions import Kind
    from raytrax.io import vtk
    from raytrax.mesh import geometry
    from raytrax.paths import AxisymmetricPath
    from raytrax.problem import Problem

    MESHES = Path(__file__).parent / "meshes"
    FIGURES = Path(__file__).parent / "figures"

    STEFAN_BOLTZMANN = 5.670374419e-8

    # The capsule is solid and unmeshed, so it is drawn as a silhouette.
    CAPSULE = "#cfd2d4"
    BLOCK_COLOURS = ("#4c72b0", "#dd8452", "#55a868", "#c44e52")
    ZOOM_BOUNDS = ((-0.04, -0.005), (0.14, 0.26))


@app.cell(hide_code=True)
def _():
    mo.md("""
    # Radiation in a re-entry shock layer

    The domain is the shock layer around a Stardust-like capsule: a 60 degree
    sphere-cone forebody, a shoulder, and a conical backshell, meshed as the
    `(x, r)` half-plane and revolved about `r = 0`. The capsule is solid, so it
    is not meshed at all -- it is the `wall` boundary, and the grey silhouette
    in every figure below.

    Four boundaries close the domain:

    - **shock** - the bow shock, treated as a black surface at `T`. It radiates
      `sigma T^4` per unit area into the layer and absorbs whatever comes back
      out.
    - **wall** - the capsule surface, cold and black. What lands here is the
      radiative load on the heat shield, and it is the answer this notebook
      exists to produce.
    - **outflow** - the plane through the base of the capsule. Gas escapes
      there, and so does radiation.
    - **axis** - the short stretch of `r = 0` between the shock and the nose.
      Nothing should ever reach it, and the energy table checks that.

    The gas is cold and grey: it absorbs at `kappa` and emits nothing, so every
    photon here started at the shock. There is no closed form to check against,
    only the energy budget, which has to balance exactly.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md("""
    ## Where the geometry comes from

    Nothing behind this is a CFD solution. The capsule carries Stardust's
    published dimensions -- 0.2202 m nose radius, a 60 degree sphere-cone
    forebody, 0.8128 m base diameter, a 0.0202 m shoulder -- and the backshell
    half-angle is then whatever lands the base where the published 0.5 m
    overall length puts it. The 10,000 K below is a round number, not a
    computed post-shock temperature.

    The shock is Billig's hyperbola. Its vertex sits one standoff, 0.02 m,
    ahead of the nose; it has 1.143 nose radii of curvature there, which is
    Billig's high-Mach limit; and it runs out to an asymptote at `shock_angle`
    to the axis. Held at the cone angle, as it is here, the far shock runs
    parallel to the forebody, so the layer thickens steadily from the
    stagnation standoff instead of pinching shut against the shoulder.

    The price of that is the domain aft of the shoulder. The shock keeps
    sweeping out at 60 degrees, so by the base plane it stands two and a half
    base radii off the axis, and most of the cells end up in an expansion
    region carrying no gradient. That is the real shape, and it is why every
    field below is drawn twice: once whole, once zoomed onto the forebody, on
    the scale of the standoff itself.

    ## What the mesh is

    Graded by distance from the stagnation point and recombined to quads: about
    ten cells across the standoff at the nose, twenty times coarser out in the
    expansion, and no clustering at the wall, because an inviscid finite-volume
    solve has no boundary layer to resolve. Triangles survive only where the
    layer is too narrow to pair cells off, which is what the element counts in
    the mesh figure show.

    `reentry()` in `examples/meshes/generate.py` builds it, and every dimension
    named above is a keyword argument there.
    """)
    return


@app.cell
def _():
    temperature = mo.ui.slider(5000, 15000, 500, value=10000, label="Shock temperature (K)")
    kappa = mo.ui.slider(0.1, 20.0, 0.1, value=2.0, label="Absorptivity kappa (1/m)")
    num_rays = mo.ui.dropdown(["200000", "1000000", "4000000"], value="1000000", label="Rays")
    mo.hstack([temperature, kappa, num_rays], justify="start", gap=2)
    return kappa, num_rays, temperature


@app.function
def solve(temperature, absorptivity, rays):
    """Trace the shock layer, returning the mesh, its ring weighting, and the deposits.

    `geom` is what the physics uses and is ring-weighted; positions stay
    in `(x, r)`, so the figures read `mesh.geometry` instead.
    """
    path = MESHES / "reentry.msh"
    if not path.exists():
        raise FileNotFoundError(f"{path} missing - run `uv run python examples/meshes/generate.py`")

    mesh = vtk.read(path)
    geom = geometry.revolve(mesh.geometry)

    scene = scene_module.build(mesh.topology, geom, mesh.groups)
    scene_module.validate(mesh.topology, geom)

    # Nothing here is a mirror, so a ray is finished the first time it
    # reaches any boundary. ABSORB rather than ESCAPE on the shock and the
    # wall only so that their deposits can be read back as a flux.
    table = interactions.build(
        mesh.groups,
        {"shock": Kind.ABSORB, "wall": Kind.ABSORB, "outflow": Kind.ESCAPE, "axis": Kind.ABSORB},
    )

    shock = np.flatnonzero(mesh.groups.face_group == mesh.groups.index("shock"))
    # Revolved, `faces.area` is already the area of the ring a face sweeps,
    # so this is the power the whole shock surface radiates.
    source = sources.build(
        cell_power=jnp.zeros(mesh.topology.num_cells),
        face_power=STEFAN_BOLTZMANN * temperature**4 * geom.faces.area[shock],
        face=shock,
    )

    problem = Problem(
        scene, 
        path=AxisymmetricPath(), 
        interactions=table, 
        absorptivity=jnp.full(mesh.topology.num_cells, absorptivity), 
        sources=source,
    )
    tally = transport.trace(jax.random.key(4), problem, num_rays=rays, max_steps=2000)
    return mesh, geom, source, problem, tally


@app.cell
def _(kappa, num_rays, temperature):
    mesh, geom, source, problem, tally = solve(temperature.value, kappa.value, int(num_rays.value))
    deposit = tally.estimate(problem)
    heating = np.asarray(deposit.cells) / geom.cells.volume
    # A face lying on the axis sweeps no ring at all, so it can never be reached.
    flux = np.asarray(deposit.faces) / np.where(geom.faces.area > 0, geom.faces.area, np.inf)
    return deposit, flux, geom, heating, mesh, problem, source, tally


@app.cell
def _(deposit, geom, mesh):
    def group_faces(name):
        """Row indices of one boundary group."""
        return np.flatnonzero(mesh.groups.face_group == mesh.groups.index(name))

    def capsule():
        """The capsule outline, closed along the axis, taken from the wall faces alone."""
        corners = mesh.geometry.vertices[mesh.topology.face_vertices[group_faces("wall")][:, :2]].reshape(-1, 2)
        # The profile runs strictly forward in x from nose to base, so sorting
        # on x walks it in order and no face-to-face chasing is needed.
        profile = corners[np.argsort(corners[:, 0])]
        return np.vstack([profile, [[profile[-1, 0], 0.0]], [[profile[0, 0], 0.0]]])

    def wall_profile():
        """Wall faces from the stagnation point back, with the surface distance to each."""
        wall = group_faces("wall")
        wall = wall[np.argsort(mesh.geometry.faces.centroid[wall, 0])]
        # In (x, r), a face is a segment and its "area" is its length.
        length = mesh.geometry.faces.area[wall]
        return wall, np.cumsum(length) - length / 2, length

    def wall_flux(bins=40):
        """Wall flux against surface distance, pooled over equal-arc bins.

        Pooling deposit and area rather than averaging per-face fluxes is what
        makes the nose usable: a face there revolves into a ring of almost no
        area, so on its own it is one or two rays over almost nothing.
        """
        wall, distance, length = wall_profile()
        edges = np.linspace(0.0, distance[-1] + length[-1] / 2, bins + 1)
        index = np.clip(np.digitize(distance, edges) - 1, 0, bins - 1)
        energy = np.bincount(index, np.asarray(deposit.faces)[wall], bins)
        area = np.bincount(index, geom.faces.area[wall], bins)
        return edges, np.divide(energy, area, out=np.full(bins, np.nan), where=area > 0)

    return capsule, group_faces, wall_flux, wall_profile


@app.cell(hide_code=True)
def _(geom, group_faces, heating, mesh, source):
    def _stats():
        counts: dict[str, int] = {}
        for index, element in enumerate(mesh.topology.block_elements):
            counts[element.name] = counts.get(element.name, 0) + int((mesh.topology.cell_block == index).sum())
        measure = mesh.geometry.cells.volume
        return mo.md(f"""
        | | |
        |---|---|
        | cells | {mesh.topology.num_cells:,} - {", ".join(f"{n:,} {t}" for t, n in counts.items())} |
        | boundary groups | {", ".join(mesh.groups.names[1:])} |
        | cell size spread | {np.sqrt(measure.max() / measure.min()):.1f}x |
        | shock area | {geom.faces.area[group_faces("shock")].sum():.4f} m2 |
        | wetted wall area | {geom.faces.area[group_faces("wall")].sum():.4f} m2 |
        | emitted | {float(source.total) / 1e6:,.1f} MW |
        | peak heating | {heating.max() / 1e9:.3f} GW/m3 |
        """)

    _stats()
    return


@app.cell
def _(capsule, mesh):
    def blocks():
        """Cell rows and their corner coordinates, one entry per element block."""
        vertices = mesh.geometry.vertices
        for index, element in enumerate(mesh.topology.block_elements):
            rows = np.flatnonzero(mesh.topology.cell_block == index)
            yield element, rows, vertices[mesh.topology.cell_vertices[rows][:, : element.num_vertices]]

    def finish(ax, zoom):
        """Lay the capsule over the cells and frame the axes."""
        outline = capsule()
        ax.fill(outline[:, 0], outline[:, 1], color=CAPSULE, zorder=3, ec="#3f3f3f", lw=0.8)
        ax.set_aspect("equal")
        ax.autoscale_view()
        if zoom:
            # The stagnation standoff is a fiftieth of the domain, so the layer
            # is only legible on its own scale.
            (xmin, ymin), (xmax, ymax) = ZOOM_BOUNDS
            ax.set_xlim(xmin, xmax)
            ax.set_ylim(ymin, ymax)
        ax.set_xlabel("x (m)")
        ax.set_ylabel("r (m)")

    def draw_mesh(ax, zoom=False):
        """Every cell outlined, flat filled by element type."""
        for (element, rows, corners), colour in zip(blocks(), BLOCK_COLOURS, strict=False):
            ax.add_collection(
                PolyCollection(corners, facecolors=colour, edgecolors="#26262699", linewidths=0.25, alpha=0.55)
            )
            ax.plot([], [], color=colour, lw=6, alpha=0.55, label=f"{len(rows):,} {element.name}")
        finish(ax, zoom)

    def draw_field(ax, values, norm, zoom=False, cmap="magma"):
        """Every cell filled with its value."""
        for _, rows, corners in blocks():
            # Edge colour follows the face: at linewidth zero VTK-style gaps still
            # show through as white hairlines between cells.
            polys = PolyCollection(corners, cmap=cmap, norm=norm, linewidths=0.2, edgecolors="face")
            polys.set_array(values[rows])
            ax.add_collection(polys)
        finish(ax, zoom)

    def domain_bounds():
        """Bounding box of the cells and the capsule outline, for viewbox alignment."""
        points = [capsule()]
        for _, _, corners in blocks():
            points.append(corners.reshape(-1, 2))
        points = np.vstack(points)
        return points.min(axis=0), points.max(axis=0)

    def _export_axes(bounds, pixels=1600):
        """A bare, unlabelled axes sized so the canvas maps exactly onto `bounds`."""
        (xmin, ymin), (xmax, ymax) = bounds
        dpi = 200
        scale = pixels / max(xmax - xmin, ymax - ymin)
        fig = plt.figure(figsize=((xmax - xmin) * scale / dpi, (ymax - ymin) * scale / dpi), dpi=dpi)
        ax = fig.add_axes((0, 0, 1, 1))
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        ax.set_aspect("equal")
        ax.axis("off")
        return fig, ax

    def export_mesh(zoom):
        """The mesh cells and capsule silhouette alone, cropped to the plotted bounds.

        No legend or axes -- only the fills, plus the legend contents as data so
        Typst can redraw the key.
        """
        bounds = ZOOM_BOUNDS if zoom else domain_bounds()
        fig, ax = _export_axes(bounds)
        legend = []
        for (element, rows, corners), colour in zip(blocks(), BLOCK_COLOURS, strict=False):
            ax.add_collection(
                PolyCollection(corners, facecolors=colour, edgecolors="#26262699", linewidths=0.25, alpha=0.55)
            )
            legend.append({"label": f"{len(rows):,} {element.name}", "colour": colour})
        outline = capsule()
        ax.fill(outline[:, 0], outline[:, 1], color=CAPSULE, zorder=3, ec="#3f3f3f", lw=0.8)
        return fig, bounds, legend

    def export_field(values, norm, zoom, cmap="magma"):
        """The field fill alone, cropped to the plotted bounds, no colorbar."""
        bounds = ZOOM_BOUNDS if zoom else domain_bounds()
        fig, ax = _export_axes(bounds)
        for _, rows, corners in blocks():
            polys = PolyCollection(corners, cmap=cmap, norm=norm, linewidths=0.2, edgecolors="face")
            polys.set_array(values[rows])
            ax.add_collection(polys)
        outline = capsule()
        ax.fill(outline[:, 0], outline[:, 1], color=CAPSULE, zorder=3, ec="#3f3f3f", lw=0.8)
        return fig, bounds

    return draw_field, draw_mesh, export_field, export_mesh


@app.cell
def _(draw_field, draw_mesh, flux, heating, mesh, wall_flux, wall_profile):
    def _panels():
        # Wide panel for the whole domain, narrow one for the forebody zoom.
        return plt.subplots(1, 2, figsize=(11, 5), width_ratios=(1.4, 1.0))

    def mesh_figure():
        """The mesh itself, whole and zoomed on the stagnation region."""
        fig, axes = _panels()
        for ax, zoom in zip(axes, (False, True), strict=True):
            draw_mesh(ax, zoom)
        axes[0].legend(fontsize=8, loc="lower right", frameon=False)
        axes[0].set_title("Shock layer, whole domain", fontsize=10)
        axes[1].set_title("Forebody, on the scale of the standoff", fontsize=10)
        fig.tight_layout()
        return fig

    def field_figure():
        """Volumetric heating, whole and zoomed."""
        fig, axes = _panels()
        norm = plt.Normalize(heating.min() / 1e9, heating.max() / 1e9)
        for ax, zoom in zip(axes, (False, True), strict=True):
            draw_field(ax, heating / 1e9, norm, zoom)
        fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap="magma"), ax=axes, label="Volumetric heating (GW/m3)")
        return fig

    def wall_figure():
        """Radiative flux along the capsule surface, nose to base."""
        wall, distance, _ = wall_profile()
        edges, binned = wall_flux()
        shoulder = distance[np.argmax(mesh.geometry.faces.centroid[wall, 1])]

        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(distance, flux[wall] / 1e6, ls="none", marker=".", ms=2.5, color="0.7", label="per face")
        ax.stairs(binned / 1e6, edges, color="C0", lw=1.6, label="pooled over the bin")
        # Framed on the pooled curve: the per-face points run off the top near
        # the nose, where the rings they revolve into carry no area.
        ax.set_ylim(0.0, 1.2 * np.nanmax(binned) / 1e6)
        ax.axvline(shoulder, color="0.4", ls=":", lw=1.0)
        ax.annotate("shoulder", (shoulder, ax.get_ylim()[1]), fontsize=8, ha="right", va="top", rotation=90)
        ax.legend(fontsize=8, frameon=False)
        ax.set_xlabel("Surface distance from the stagnation point (m)")
        ax.set_ylabel("Radiative wall flux (MW/m2)")
        fig.tight_layout()
        return fig

    return field_figure, mesh_figure, wall_figure


@app.cell(hide_code=True)
def _(mesh_figure):
    mesh_figure()
    return


@app.cell(hide_code=True)
def _(field_figure):
    field_figure()
    return


@app.cell(hide_code=True)
def _(wall_figure):
    wall_figure()
    return


@app.cell(hide_code=True)
def _():
    mo.md("""
    The forebody load is nearly flat and then falls off a cliff at the
    shoulder: past it the wall turns away from the shock and sees much less of
    it.

    The grey scatter near the nose is sampling, not physics. A face there
    revolves into a ring of almost no area, so hardly any rays land on it and
    its own flux is one or two deposits over almost nothing. The blue curve is
    the estimator to read.

    ### The binning

    Forty bins of equal *arc length* along the wetted profile, measured from
    the stagnation point, and each bin's flux is the energy its faces caught
    divided by the area those faces revolve into. Three choices in that, each
    of which could have gone the other way:

    - **Pooled, not averaged.** The bin value is `sum(energy) / sum(area)`,
      not the mean of the per-face fluxes it contains. Averaging the fluxes
      would weight a hairline ring at the nose exactly as heavily as a broad
      one further out, which is what puts the grey points where they are. This
      way a face contributes in proportion to the area it actually presents,
      and the bin means what it says: the mean flux over that strip of wall.
    - **Arc length, not area.** Equal-area bins would crowd the shoulder,
      where the rings are widest, and swallow the whole nose in one. Equal arc
      keeps the resolution uniform along the surface, which is what it takes
      to read *where* the cliff at the shoulder falls.
    - **Forty of them.** Fine enough that the drop at the shoulder lands in a
      single bin rather than being smeared across several, and coarse enough
      that the first bin, always the worst-sampled one, still gathers a few
      hundred rays at a million launched. `wall_flux(bins=...)` is the knob.
      The plateau sits at the same level anywhere from twenty bins to a
      hundred; past that the first bin is down to tens of rays, and since
      there are only a couple of hundred wall faces in the whole profile,
      bins start coming out holding no face at all.
    """)
    return


@app.cell(hide_code=True)
def _(deposit, flux, group_faces, num_rays, problem, source, tally):
    def _budget():
        emitted = float(source.total)
        # A terminating ray is emptied into the deposit, so a lane still holding
        # anything is one the step cap caught part-way.
        escaped = float(deposit.escaped)
        in_flight = float(tally.in_flight(problem))
        faces = np.asarray(deposit.faces)
        gas = float(deposit.cells.sum())
        total = gas + float(deposit.faces.sum()) + escaped + in_flight
        wall = group_faces("wall")

        def row(name, value):
            return f"| {name} | {value / 1e6:,.1f} MW | {value / emitted:.4f} |"

        # A ray only reaches r = 0 with no angular momentum at all, so anything landing on
        # the axis group is a leak through the symmetry axis.
        return mo.md(f"""
        | energy | | fraction |
        |---|---|---|
        | emitted at the shock | {emitted / 1e6:,.1f} MW | 1.0000 |
        {row("absorbed in the gas", gas)}
        {row("absorbed at the wall", faces[wall].sum())}
        {row("back out through the shock", faces[group_faces("shock")].sum())}
        {row("out through the outflow", escaped)}
        {row("through the axis (must be zero)", faces[group_faces("axis")].sum())}
        {row("still in flight", in_flight)}
        | **accounted** | **{total / 1e6:,.1f} MW** | error {abs(total / emitted - 1):.2e} |

        Peak wall flux {flux[wall].max() / 1e6:,.1f} MW/m2, with {int(tally.rays.alive.sum()):,} of
        {int(num_rays.value):,} rays left alive at the step limit.
        """)

    _budget()
    return


@app.cell(hide_code=True)
def _():
    write = mo.ui.run_button(label="Write every figure and CSV")
    mo.md(f"""
    ## Export

    Every export is meant to be laid out in Typst, not read on its own. Line
    and scatter data go out as CSV. The mesh and heating images hold only the
    fills -- no axes, labels, legend, or colorbar -- so each comes with a JSON
    sidecar giving the `(xmin, ymin, xmax, ymax)` viewbox the image was cropped
    to, plus the colorbar range or legend contents where those apply.

    {write}
    """)
    return (write,)


@app.cell(hide_code=True)
def _(
    export_field,
    export_mesh,
    flux,
    heating,
    mesh,
    wall_flux,
    wall_profile,
    write,
):
    def _save_csv(name, columns, header):
        path = FIGURES / name
        np.savetxt(path, np.column_stack(columns), delimiter=",", header=header, comments="", fmt="%.8g")
        return path

    def _save_json(name, data):
        path = FIGURES / name
        path.write_text(json.dumps(data, indent=2))
        return path

    def _save_image(stem, fig, bounds, **metadata):
        (xmin, ymin), (xmax, ymax) = bounds
        image = FIGURES / f"{stem}.png"
        fig.savefig(image, transparent=True)
        plt.close(fig)
        sidecar = _save_json(f"{stem}.json", {"xmin": xmin, "ymin": ymin, "xmax": xmax, "ymax": ymax, **metadata})
        return image, sidecar

    def _export():
        if not write.value:
            return mo.md("*Not written yet.*")
        FIGURES.mkdir(exist_ok=True)

        written = []
        heating_min, heating_max = float(heating.min() / 1e9), float(heating.max() / 1e9)
        norm = plt.Normalize(heating_min, heating_max)
        for zoom, suffix in ((False, "whole"), (True, "zoom")):
            fig, bounds, legend = export_mesh(zoom)
            written.extend(_save_image(f"reentry_mesh_{suffix}", fig, bounds, legend=legend))

            fig, bounds = export_field(heating / 1e9, norm, zoom)
            colorbar = {"min": heating_min, "max": heating_max, "cmap": "magma", "label": "GW/m3"}
            written.extend(_save_image(f"reentry_heating_{suffix}", fig, bounds, colorbar=colorbar))

        wall, distance, _ = wall_profile()
        centroid = mesh.geometry.faces.centroid[wall]
        written.append(
            _save_csv("reentry_wall_flux.csv", (distance, centroid[:, 0], centroid[:, 1], flux[wall]), "s,x,r,flux")
        )
        edges, binned = wall_flux()
        written.append(_save_csv("reentry_wall_flux_binned.csv", ((edges[:-1] + edges[1:]) / 2, binned), "s,flux"))
        shoulder = distance[np.argmax(mesh.geometry.faces.centroid[wall, 1])]
        written.append(_save_json("reentry_wall_flux.json", {"shoulder": float(shoulder)}))

        cells = mesh.geometry.cells.centroid
        written.append(_save_csv("reentry_heating.csv", (cells[:, 0], cells[:, 1], heating), "x,r,heating"))
        return mo.md("Wrote:\n\n" + "\n".join(f"- `{path.name}`" for path in written))

    _export()
    return


if __name__ == "__main__":
    app.run()
