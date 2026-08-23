import marimo

__generated_with = "0.24.0"
app = marimo.App(width="medium")

with app.setup:
    import itertools
    from pathlib import Path

    import jax
    import jax.numpy as jnp
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.collections import LineCollection, PolyCollection
    from scipy.special import expn

    from raytrax import interactions, sources, transport
    from raytrax import scene as scene_module
    from raytrax.interactions import Kind
    from raytrax.io import vtk
    from raytrax.mesh import geometry
    from raytrax.paths import AxisymmetricPath, LinearPath

    MESHES = Path(__file__).parent / "meshes"


@app.cell(hide_code=True)
def _():
    mo.md("""
    # Ray tracing a participating medium

    A cold absorbing gas is heated by a hot wall at the low-x end. Both
    domains use the same setup, so the only thing that changes is the mesh.

    - **cylinder** — 3D tetrahedra. With the lateral surface mirrored the
      domain is effectively an infinite slab, so the axial heating profile
      has a closed form to check against.
    - **plane** — 2D, deliberately **mixed**: quads on the lower half,
      triangles on the upper, sharing an interior face.

    - **meridian** — the same solid as the cylinder, meshed as its 2D
      meridional half-plane and read as **axisymmetric**.

    A 2D mesh can be read two ways. **Planar** extrudes it infinitely out of
    plane; only the in-plane part of a direction is stored, so `|tangent| < 1`
    and the distance flown exceeds the in-plane chord. **Axisymmetric** revolves
    it, and a straight 3D ray then projects to a *hyperbola* rather than a line.
    Either way `travel` is the true 3D length, so all three domains are checked
    against the same closed form.
    """)
    return


@app.cell
def _():
    domain = mo.ui.dropdown(["cylinder", "plane", "meridian"], value="cylinder", label="Domain")
    kappa = mo.ui.slider(0.1, 5.0, 0.1, value=1.0, label="Absorptivity κ")
    num_rays = mo.ui.dropdown(["50000", "200000", "1000000"], value="200000", label="Rays")
    mirror = mo.ui.checkbox(True, label="Mirror the lateral boundary")
    mo.hstack([domain, kappa, num_rays, mirror], justify="start", gap=2)
    return domain, kappa, mirror, num_rays


@app.cell
def _(domain):
    path = MESHES / f"{domain.value}.msh"
    if not path.exists():
        raise FileNotFoundError(f"{path} missing — run `uv run python examples/meshes/generate.py`")

    mesh = vtk.read(path)
    # Revolving reweights cells as rings, which the axisymmetric path assumes.
    axisymmetric = domain.value == "meridian"
    geom = geometry.revolve(mesh.geometry) if axisymmetric else mesh.geometry
    trace_path = AxisymmetricPath() if axisymmetric else LinearPath()

    scene = scene_module.build(mesh.topology, geom, mesh.groups)
    scene_module.validate(mesh.topology, geom)
    return geom, mesh, scene, trace_path


@app.cell(hide_code=True)
def _(geom, mesh):
    _counts = {
        element.name: int((mesh.topology.cell_block == index).sum())
        for index, element in enumerate(mesh.topology.block_elements)
    }
    mo.md(f"""
    | | |
    |---|---|
    | dimension | {mesh.topology.dim} |
    | cells | {mesh.topology.num_cells:,} — {", ".join(f"{n:,} {t}" for t, n in _counts.items())} |
    | faces | {mesh.topology.num_faces:,} |
    | face slots per cell | {mesh.topology.max_faces} |
    | boundary groups | {", ".join(mesh.groups.names[1:])} |
    | total volume | {geom.cells.volume.sum():.6f} |
    """)
    return


@app.cell
def _(geom, mesh, mirror):
    # A black hot wall, a black cold wall, and a lateral surface that either
    # mirrors rays back in or lets them escape.
    lateral = Kind.SPECULAR if mirror.value else Kind.ESCAPE
    table = interactions.build(mesh.groups, {"hot_end": Kind.ABSORB, "wall": Kind.ABSORB, "lateral": lateral})

    # Cold gas; the hot end cap emits with sigma T^4 = 1 per unit area.
    hot = np.flatnonzero(mesh.groups.face_group == mesh.groups.index("hot_end"))
    source = sources.build(
        cell_power=jnp.zeros(mesh.topology.num_cells),
        face_power=geom.faces.area[hot],
        face=hot,
    )
    return hot, source, table


@app.cell
def _(geom, kappa, mesh, num_rays, scene, source, table, trace_path):
    _rays = int(num_rays.value)
    _path = trace_path
    key_launch, key_march = jax.random.split(jax.random.key(4))

    # Sampling explicitly rather than through `transport.trace` keeps the launch
    # states around for the ray-path plot below.
    launch_state, launch_cell = jax.vmap(sources.sample, in_axes=(0, None, None, None))(
        jax.random.split(key_launch, _rays), source, scene, _path
    )
    initial = transport.Rays(
        cell=launch_cell,
        state=launch_state,
        energy=jnp.full(_rays, source.total / _rays),
        alive=jnp.ones(_rays, dtype=bool),
    )
    absorptivity = jnp.full(mesh.topology.num_cells, kappa.value)

    final, deposit = transport.march(key_march, initial, scene, _path, table, absorptivity, max_steps=4000)
    heating = np.asarray(deposit.cells) / geom.cells.volume
    return deposit, final, heating, initial


@app.cell(hide_code=True)
def _(deposit, final, initial, mesh, source):
    # A ray absorbed at a wall has already handed over its energy, so whatever a
    # dead ray still carries left the domain.
    _escaped = float(jnp.where(final.alive, 0.0, final.energy).sum())
    _flight = float(jnp.where(final.alive, final.energy, 0.0).sum())
    _total = float(deposit.cells.sum() + deposit.faces.sum()) + _escaped + _flight

    # A ray only reaches r = 0 with no swirl at all, so anything landing on the
    # axis group is a leak through the symmetry plane.
    _axis = ""
    if "axis" in mesh.groups.names:
        _faces = np.flatnonzero(mesh.groups.face_group == mesh.groups.index("axis"))
        _leak = float(np.asarray(deposit.faces)[_faces].sum()) / float(source.total)
        _axis = f"| through the axis | {_leak:.2e} (must be zero) |\n"

    mo.md(f"""
    | energy | |
    |---|---|
    | emitted | {float(source.total):.6f} |
    | absorbed in gas | {float(deposit.cells.sum()):.6f} |
    | absorbed at walls | {float(deposit.faces.sum()):.6f} |
    | escaped | {_escaped:.6f} |
    {_axis}
    | still in flight | {_flight:.6f} |
    | **accounted** | **{_total:.6f}** (error {abs(_total / float(source.total) - 1):.2e}) |

    {int(final.alive.sum()):,} of {initial.energy.size:,} rays ran out of steps. A mirrored
    box keeps a few rays bouncing long after they carry anything, so judge
    convergence by the accounting error above rather than by that count --
    raising `max_steps` here shrinks the count without moving the answer.
    """)
    return


@app.cell(hide_code=True)
def _(geom, heating, kappa, mesh, mirror):
    def _axial():
        fig, ax = plt.subplots(figsize=(7, 4))
        x = mesh.geometry.cells.centroid[:, 0]
        lo, hi = mesh.geometry.vertices[:, 0].min(), mesh.geometry.vertices[:, 0].max()

        bins = np.linspace(lo, hi, 26)
        index = np.clip(np.digitize(x, bins) - 1, 0, bins.size - 2)
        energy = np.bincount(index, heating * geom.cells.volume, bins.size - 1)
        volume = np.bincount(index, geom.cells.volume, bins.size - 1)
        ax.stairs(energy / volume, bins, label="Monte-Carlo", lw=2)

        if mirror.value:
            # Mirrored sides make this an infinite slab in every mode: planar
            # extrusion and revolution are both just the slab's other two
            # directions. The
            # cell-averaged heating over [a, b] is 2 (E3(ka) - E3(kb)) / (b - a).
            near, far = kappa.value * (bins[:-1] - lo), kappa.value * (bins[1:] - lo)
            exact = 2 * (expn(3, near) - expn(3, far)) / (bins[1:] - bins[:-1])
            ax.stairs(exact, bins, label="Tangent slab", ls="--", color="C1", lw=2)

        ax.set_xlabel("x")
        ax.set_ylabel("Volumetric heating")
        ax.legend()
        return fig

    _axial()
    return


@app.cell(hide_code=True)
def _(heating, kappa, mesh):
    def _radial():
        if mesh.topology.dim != 3:
            return mo.md("*Radial shells apply to the 3D cylinder only.*")

        fig, ax = plt.subplots(figsize=(7, 4))
        centroid = mesh.geometry.cells.centroid
        x = centroid[:, 0]
        radius = np.linalg.norm(centroid[:, 1:], axis=-1)
        lo = mesh.geometry.vertices[:, 0].min()

        x_bins = np.linspace(lo, mesh.geometry.vertices[:, 0].max(), 5)
        r_bins = np.linspace(0.0, np.linalg.norm(mesh.geometry.vertices[:, 1:], axis=-1).max(), 31)
        xi = np.clip(np.digitize(x, x_bins) - 1, 0, x_bins.size - 2)
        ri = np.clip(np.digitize(radius, r_bins) - 1, 0, r_bins.size - 2)
        flat = xi * (r_bins.size - 1) + ri
        shape = (x_bins.size - 1, r_bins.size - 1)

        energy = np.bincount(flat, heating * mesh.geometry.cells.volume, shape[0] * shape[1]).reshape(shape)
        volume = np.bincount(flat, mesh.geometry.cells.volume, shape[0] * shape[1]).reshape(shape)

        for i, (a, b) in enumerate(itertools.pairwise(x_bins)):
            ax.stairs(energy[i] / volume[i], r_bins, color=f"C{i}", label=f"${a:+.2f} < x < {b:+.2f}$")
            # The slab solution is radially uniform, so it is a flat reference.
            depth = kappa.value * (np.array([a, b]) - lo)
            ax.axhline(2 * (expn(3, depth[0]) - expn(3, depth[1])) / (b - a), color=f"C{i}", ls="--", lw=0.8)

        ax.set_xlabel("r")
        ax.set_ylabel("Volumetric heating")
        ax.legend(fontsize=8)
        return fig

    _radial()
    return


@app.cell(hide_code=True)
def _(heating, mesh):
    def _field():
        if mesh.topology.dim != 2:
            return mo.md("*The cell-by-cell field view is drawn for 2D domains only.*")

        fig, ax = plt.subplots(figsize=(7, 4))
        topology, vertices = mesh.topology, mesh.geometry.vertices
        norm = plt.Normalize(heating.min(), heating.max())

        # One collection per block, so tris and quads can differ in width.
        for index, element in enumerate(topology.block_elements):
            rows = np.flatnonzero(topology.cell_block == index)
            corners = vertices[topology.cell_vertices[rows][:, : element.num_vertices]]
            polys = PolyCollection(corners, cmap="magma", norm=norm, linewidths=0)
            polys.set_array(heating[rows])
            ax.add_collection(polys)

        ax.set_aspect("equal")
        ax.autoscale_view()
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap="magma"), ax=ax, label="Volumetric heating")
        return fig

    _field()
    return


@app.cell(hide_code=True)
def _(final, initial, mesh):
    def _paths():
        if mesh.topology.dim != 2:
            return mo.md("*Ray paths are drawn for 2D domains only.*")

        fig, ax = plt.subplots(figsize=(7, 4))
        vertices = mesh.geometry.vertices

        edges = LineCollection(vertices[mesh.topology.face_vertices[:, :2]], linewidths=0.15, colors="0.75")
        ax.add_collection(edges)

        # Start-to-finish chords. Three reasons the colour exceeds the length
        # drawn: a mirrored ray flew a longer, folded path, the part of the
        # direction that points out of plane is not drawable at all, and an
        # axisymmetric ray followed a hyperbola rather than this chord.
        show = slice(0, 200)
        chords = np.stack([initial.state.position[show], final.state.position[show]], axis=1)
        paths = LineCollection(list(chords), cmap="viridis", linewidths=0.6)
        paths.set_array(np.asarray(final.state.travel[show]))
        ax.add_collection(paths)

        ax.set_aspect("equal")
        ax.autoscale_view()
        fig.colorbar(paths, ax=ax, label="Distance flown")
        return fig

    _paths()
    return


@app.cell(hide_code=True)
def _(deposit, domain, geom, heating, hot, mesh):
    _out = MESHES / f"{domain.value}_heating.vtu"
    _face = np.asarray(deposit.faces) / np.where(geom.faces.area > 0, geom.faces.area, np.inf)
    vtk.write(_out, mesh, {"heating": heating})
    mo.md(f"Wrote `{_out.name}` — cell heating, openable in ParaView. Peak wall flux {_face[hot].max():.4f}.")
    return


if __name__ == "__main__":
    app.run()
