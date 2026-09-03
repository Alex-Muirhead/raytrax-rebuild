# Architecture

Where the code lives and what must hold across it. `DECISIONS.md` states the
argument behind each invariant; this file states the invariant.

## Layout

The hard split is **host vs device**, and it is what keeps mixed element types
manageable:

- `elements.py` - element/face tables. One entry per first-order VTK cell.
- `mesh/` - pure NumPy, no JAX. Topology, geometry, boundary groups.
- `scene.py` - the boundary. `build()` flattens a mesh into padded rectangular
  arrays; everything downstream is JAX.
- `paths.py`, `medium.py`, `interactions.py`, `sources.py`, `transport.py` - JAX.

## Physics invariants

Breaking one of these gives a wrong answer that still runs, so they come first.

- **The project runs in float64.** This is physics, not graphics: the answer is
  a number someone will compare against a measurement, and float32's seven
  digits are not enough to stand behind.
  `raytrax/__init__.py` sets it as an import side effect, before any array
  exists.
- **Energy is conserved at every step.** `transport.step` splits the energy a
  ray carries into `1 - exp(-kappa s)` deposited in the cell and the rest
  carried on, then at a boundary into `energy * absorbed` deposited on the face
  and the rest carried on. Nothing is created and a ray's energy only falls, so
  deposits plus what is still in flight plus what escaped equal what was
  launched.
- **Every launched ray carries the same quantum, `sources.total / num_rays`.**
  The deposit map is then in source power units whatever the lane count, and
  refilling a dead lane mid-march cannot bias it. All rays must account for
  their quantum via deposition either in cells or boundaries the time they
  terminate.
- **Cell and face emission are in the same units, so one draw over
  `Sources.cumulative` picks between them.** Volume emission is isotropic;
  surface emission is cosine-weighted about the inward normal.
- **`random.simplex` is uniform over the simplex.** Sorted uniforms cut [0, 1]
  into spans that are already Dirichlet(1), and the gaps telescope to exactly
  one, so a sampled point is an exact convex combination of the corners. The
  proof is `docs/simplex.typ`.

## Geometry invariants

- **Directions are always drawn in 3D, whatever the mesh.** In 2D only the
  in-plane part is stored, so `tangent` is short of unit length and the deficit
  stands for the out-of-plane component. `travel` is therefore the true 3D path
  length in every mode, and `exp(-kappa s)` needs no per-mode correction.
- **The crossing parameter is a distance only for `LinearPath`.** `advance`
  returns the 3D length flown alongside the new state; use it. Both modes have
  the length exactly, so differencing `travel` across a step only recovers it by
  cancellation, losing the low bits once `travel` grows.
- **The swirl `r |d_z|` is conserved along a ray and through a bounce, and is
  carried in the state rather than recomputed.** A wall leaves `r` alone and
  mirroring preserves `|tangent|`. Recovering it from `1 - |tangent|^2` cancels
  once the closest approach falls below `sqrt(eps)` of the radius.
- **A revolved mesh is stored `(x, r)` in columns 0 and 1, with `r >= 0`.**
- **`AxisymmetricPath` needs geometry passed through `geometry.revolve`.**
  Without it the cells are still Cartesian and the energy is wrong.
- **Face orientation comes from canonical winding, not permutation parity.**
  Two cells sharing a face wind it oppositely; `canonical_cycle` starts at the
  lowest vertex id and walks the lexicographically smaller direction, so they
  agree on the key and differ on one bit. Parity cannot do this for quads (both
  windings can sort with the same parity), which is why hex meshes need this.
- **Faces must be planar.** `scene.validate` rejects warped quads; they break
  the half-space cell model and rays leak.
- **`cells.normal` is signed outward per cell; `faces.normal` is the raw one.**
  The device tables are grouped by leading index, so the two never collide, and
  `Scene.__check_init__` is what checks they agree on `ndim`: the shape hook
  only sees inside one table at a time, and only under test.
- **`MeshTopology` names a field for what one row holds.** `cell_faces` and
  `face_cells` are duals; `cell_face_valid` and `face_num_verts` are derived
  from the -1 padding rather than stored.

## JAX and device notes

- **Cells are padded to `max_faces`, derived from the element types present.**
  A tet-only mesh pads to 4, a hex mesh to 6. Padded slots hold a zero normal,
  which `crossing` rejects for free, and an EXTERIOR neighbour so a degenerate
  lookup escapes rather than walking somewhere wrong.
- **`Interactions.kind` is static, and `apply` builds the switch from the kinds
  present.** Under `vmap` a `lax.switch` evaluates every branch it is given, so
  a branch no group asks for is not free: a dead DIFFUSE entry costs every ray a
  Lambertian draw per step. Changing a kind recompiles.
- **`random.simplex` sorts with a comparator network, not `jnp.sort`.** Under
  `vmap`, `jnp.sort` becomes a segmented radix sort over the whole batch: for the
  one to three elements a simplex draw needs, that costs about 2000 times a
  network of at most three comparisons. The network returns the same doubles.
- **The energy scatter must stay outside the `vmap`.** Inside it, XLA
  materialises a per-ray copy of the accumulator and memory explodes.
