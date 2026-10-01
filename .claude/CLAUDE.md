# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

# raytrax

Monte-Carlo volumetric radiation transport by ray-marching an unstructured mesh,
on the GPU via JAX.

Module layout and the invariants a change must not violate are in
`.claude/ARCHITECTURE.md`. Read it before changing anything in `mesh/`,
`scene.py` or `paths.py`, and add to it when a new invariant appears.

## Commands

```
uv run pytest                                   # everything
JAX_PLATFORMS=cpu uv run pytest tests/unit      # one function at a time, about 15 s
uv run pytest tests/unit/test_paths.py -k angular_momentum   # one file, one selection
uv run ruff format . && uv run ruff check --fix .
uv run pyrefly check                            # strict over src/, relaxed in tests/examples/scripts
```

The suite is split by what a test runs. `tests/unit` calls one function
(`sample`, `step`, a path's crossing, the mesh builders) and needs no march.
`tests/integration` runs `march` or `trace` end to end, checking conservation
and bookkeeping. `tests/verification` compares a full run against an analytic
solution. The small meshes and `uniform_boundary` are shared from
`tests/meshes.py`, imported as `tests.meshes`.

Nearly all of the suite's time is XLA compilation, one per mesh shape and per
static argument, and `JAX_PLATFORMS=cpu` compiles faster than the GPU for these
small meshes. Wrap a `jax.vmap` in a test in `eqx.filter_jit`: run eagerly, it
dispatches op by op and costs seconds.

`tests/conftest.py` installs the jaxtyping/beartype import hook, so shape
annotations are only enforced under pytest. A shape bug can therefore pass in a
script and fail in a test.

Examples are marimo notebooks, one directory each under `examples/`, one per
thesis section. Each writes `data/` and compiles its `figures.typ` into
`figures/`; neither is committed. Code they share is the `raytrax-examples`
workspace member in `examples/src/raytrax_examples`, installed by the dev group
and not part of `raytrax`. Meshes are generated on first use into
`examples/meshes/`. `examples/README.md` is the map.

```
uv run python examples/reentry/reentry.py       # one example, headless: data/ and figures/
uv run marimo edit examples/reentry/reentry.py  # the same, interactively
uv run python -m raytrax_examples.meshes        # regenerate every mesh
uv run python -m raytrax_examples.benchmark --case slab_planar --lanes 65536
uv run --group profiling python scripts/run_profile.py --case reentry   # stages and kernels of a whole run
```

`uv sync --group visualisation` adds pyvista for the 3D renders. A plain
`uv sync` removes the `visualisation` and `profiling` groups; `uv sync
--all-groups` keeps everything.

`JAX_PLATFORMS=cpu` runs the same code on the host. `.envrc` sets
`XLA_PYTHON_CLIENT_PREALLOCATE=false`.

## The pipeline

A run is a fixed sequence, and the modules only make sense in it. Host-side
NumPy up to `scene.build`, JAX after it:

1. `io.vtk.read(path)` gives a `Mesh` holding `topology`, `geometry`, `groups`
   and the original meshio object. It keeps only cells of the highest
   dimension; lower-dimensional blocks become named boundary groups.
2. For an axisymmetric reading of a 2D mesh, `mesh.geometry.revolve(geometry)`
   reweights cell volumes and face areas into rings. Skipping it and still
   using `AxisymmetricPath` runs and gives the wrong energy.
3. `scene.build(topology, geometry, groups)` flattens everything into padded
   rectangular device arrays. `scene.validate` is the separate planarity check.
4. `interactions.build(groups, kinds)` maps group names to `Kind`. Groups left
   out escape. `Kind` is static, so the set of kinds present sets what
   `interactions.apply` traces, and changing one recompiles.
5. `sources.build(cell_power, face_power, face)` builds the emission cdf; cell
   and face power share units so one draw picks between them.
6. `medium.Absorptivity.coefficient()` gives the per-cell array `Problem` wants,
   from a constant, a mesh field or a temperature model.
7. `problem.Problem(scene, path, interactions, absorptivity, sources)` is what
   is simulated.
8. `transport.trace(key, problem, num_rays)` returns a `Tally`, and
   `tally.estimate(problem)` the `Deposit`: `cells`, `faces` and the scalar
   `escaped`, which with `tally.in_flight(problem)` equal `sources.total`.
   `tally.extend(problem, num_rays)` adds `num_rays` more rays to the same
   tally.

`path` is `LinearPath` or `AxisymmetricPath` and must match what step 2 did.
Path subclasses are stateless: every method is a classmethod, and per-ray
constants live in the `RayState` that `launch` builds.

`trace` is `launch_and_march` decorated with `jit_tools.autosized("lanes")`:
leaving `lanes` out takes the width from `jit_tools.device_width` and narrows it
again on an overrun, while a lane count passed in is marched as given and lets
the overrun raise. `launch_and_march` makes an empty `Tally` and calls
the one compiled loop, `Tally.extend`. `march` takes a fixed
`steps` and never launches or stops; `extend` refills dead lanes, calls it for
`refill_interval` steps, and stops between those chunks once every ray is
spent. Each ray launches with energy 1, and a ray at or below `extinction` is
stopped and emptied where it stands. `jit_tools.lane_capacity` is the
separate memory ceiling, and is no longer on the default path.

## Documents

The three files alongside this one are written for whoever is changing the code.
`README.md` is the one for a reader of the project, and deliberately does not
point at them:

- `.claude/ARCHITECTURE.md` - layout and the invariants. Read before touching
  `mesh/`, `scene.py` or `paths.py`.
- `.claude/DECISIONS.md` - why each choice was made and what it was chosen over,
  newest first. Add an entry when a design question is settled.
- `.claude/KNOWN_ISSUES.md` - bugs found and not yet fixed, with repro detail.
  Remove an entry when a test covers it.
- `docs/*.typ` - the maths that ASCII cannot carry, chiefly `hyperbolic.typ` for
  the axisymmetric crossing.

## Conventions

- Package manager **uv** - `uv run pytest`, `uv run python ...`
- Format and organise imports with **ruff**; type-check with **pyrefly**
- Version control is **jj**, not git (`jj st`, `jj diff`, `jj describe`). The
  working copy is already a commit, so check `jj st` before the first edit of a
  new feature: reuse the current change if it is empty, otherwise `jj new`
  first. Editing first folds the feature into the previous change.
- Strict ASCII in every file, prose included. No em-dashes, no Greek, no
  symbols. Maths that needs them goes in a Typst file under `docs/`.
- Docstrings say *what*, not *how*. Attempt to describe the responsibility of
  the function in a single line. Add a longer description only where a caller
  would get it wrong without one; the aim is that the docstring costs less to read than
  the code it describes. Never how it works or why it was designed that way -
  that goes in `.claude/DECISIONS.md` or `docs/`, and real invariants go in
  `.claude/ARCHITECTURE.md`.
- Docstrings should also document *what* each parameter is, not *how* is it used.
  Attempt to describe the responsibility of each parameter in a single line.
- Docstrings use the NumPy format: `Parameters`, `Returns` and `Raises` for
  functions, `Attributes` for dataclasses, `eqx.Module`s, `NamedTuple`s and
  enums, one entry per field. Names only, no types; the annotations carry them.
  Field descriptions go in `Attributes`, not trailing comments. An `@override`
  inherits its docstring and a nested helper needs none.
- Write docstrings in ASD-STE100 Simplified Technical English where possible:
  present tense, active voice, short sentences, one meaning per word.
- Comment only what the code cannot say for itself, chiefly framework oddities,
  and say *why* the code deviates from idiom.
- Commit messages are one-liners. Add a body only to record a fact the diff
  cannot show, and keep it to a sentence or two.
- `.claude/DECISIONS.md` records why a design was chosen and what it was chosen
  over. `.claude/ARCHITECTURE.md` states the invariant; that one states the
  argument.
- Prefer invariance tests over piles of hardcoded cases; if a specific case is
  tested, say what it is looking for.
