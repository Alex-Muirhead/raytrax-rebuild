# raytrax

Monte-Carlo volumetric radiation transport by ray-marching an unstructured mesh,
on the GPU via JAX.

## Layout

The hard split is **host vs device**, and it is what keeps mixed element types
manageable:

- `elements.py` - element/face tables. One entry per first-order VTK cell.
- `mesh/` - pure NumPy, no JAX. Topology, geometry, boundary groups.
- `scene.py` - the boundary. `build()` flattens a mesh into padded rectangular
  arrays; everything downstream is JAX.
- `paths.py`, `medium.py`, `interactions.py`, `sources.py`, `transport.py` - JAX.

## Invariants worth knowing

- **Face orientation comes from canonical winding, not permutation parity.**
  Two cells sharing a face wind it oppositely; `canonical_cycle` starts at the
  lowest vertex id and walks the lexicographically smaller direction, so they
  agree on the key and differ on one bit. Parity cannot do this for quads (both
  windings can sort with the same parity), which is why hex meshes need this.
- **Cells are padded to `max_faces`, derived from the element types present.**
  A tet-only mesh pads to 4, a hex mesh to 6. Padded slots hold a zero normal,
  which `crossing` rejects for free, and an EXTERIOR neighbour so a degenerate
  lookup escapes rather than walking somewhere wrong.
- **The energy scatter must stay outside the `vmap`.** Inside it, XLA
  materialises a per-ray copy of the accumulator and memory explodes.
- **`cells.normal` is signed outward per cell; `faces.normal` is the raw one.**
  The device tables are grouped by leading index, so the two never collide, and
  `Scene.__check_init__` is what checks they agree on `ndim`: the shape hook
  only sees inside one table at a time, and only under test.
- **`MeshTopology` names a field for what one row holds.** `cell_faces` and
  `face_cells` are duals; `cell_face_valid` and `face_num_verts` are derived
  from the -1 padding rather than stored.
- **Faces must be planar.** `scene.validate` rejects warped quads; they break
  the half-space cell model and rays leak.
- **A revolved mesh is stored `(x, r)` in columns 0 and 1, with `r >= 0`.**
- **`AxisymmetricPath` needs geometry passed through `geometry.revolve`.**
  Without it the cells are still Cartesian and the energy is wrong.
- **The crossing parameter is a distance only for `LinearPath`.** `advance`
  returns the 3D length flown alongside the new state; use it. Both modes have
  the length exactly, so differencing `travel` across a step only recovers it by
  cancellation, losing the low bits once `travel` grows.

## Conventions

- Package manager **uv** - `uv run pytest`, `uv run python ...`
- Format and organise imports with **ruff**; type-check with **pyrefly**
- Version control is **jj**, not git (`jj st`, `jj diff`, `jj describe`). The
  working copy is already a commit, so check `jj st` before the first edit of a
  new feature: reuse the current change if it is empty, otherwise `jj new`
  first. Editing first folds the feature into the previous change.
- Strict ASCII in every file, prose included. No em-dashes, no Greek, no
  symbols. Maths that needs them goes in a Typst file under `docs/`.
- Docstrings say *what*, in one line. Add a caveat only where a caller would get
  it wrong without one; the aim is that the docstring costs less to read than
  the code it describes. Never how it works or why it was designed that way -
  that goes in `DECISIONS.md` or `docs/`, and real invariants go in this file.
- Comment only what the code cannot say for itself, chiefly framework oddities,
  and say *why* the code deviates from idiom.
- Commit messages are one-liners. Add a body only to record a fact the diff
  cannot show, and keep it to a sentence or two.
- `DECISIONS.md` records why a design was chosen and what it was chosen over.
  This file states the invariant; that one states the argument.
- Prefer invariance tests over piles of hardcoded cases; if a specific case is
  tested, say what it is looking for.
