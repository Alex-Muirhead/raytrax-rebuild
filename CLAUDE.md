# raytrax

Monte-Carlo volumetric radiation transport by ray-marching an unstructured mesh,
on the GPU via JAX.

## Layout

The hard split is **host vs device**, and it is what keeps mixed element types
manageable:

- `elements.py` — element/face tables. One entry per first-order VTK cell.
- `mesh/` — pure NumPy, no JAX. Topology, geometry, boundary groups.
- `scene.py` — the boundary. `build()` flattens a mesh into padded rectangular
  arrays; everything downstream is JAX.
- `paths.py`, `medium.py`, `interactions.py`, `sources.py`, `transport.py` — JAX.

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
- **Faces must be planar.** `scene.validate` rejects warped quads; they break
  the half-space cell model and rays leak.

## Conventions

- Package manager **uv** — `uv run pytest`, `uv run python ...`
- Format and organise imports with **ruff**; type-check with **pyrefly**
- Version control is **jj**, not git (`jj st`, `jj diff`, `jj describe`)
- Keep docstrings and comments short. Prefer invariance tests over piles of
  hardcoded cases; if a specific case is tested, say what it is looking for.
