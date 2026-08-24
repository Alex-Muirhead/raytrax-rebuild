# Decisions

Why the design is the way it is, and what each choice was chosen over. Newest
first.

`CLAUDE.md` states the invariants a reader must not violate. This file states
the argument behind them, which the conventions deliberately keep out of
docstrings, comments and commit messages. Keep entries to the four lines below;
anything longer belongs in `docs/`.

## Path methods are classmethods, and a mode holds no state

`Make the Path interface stateless`, 2026-08-24

**Why:** No mode has instance state, so `self` advertised a statefulness that
does not exist. `spatial_dim` was the last thing reached through the instance
and is constant 3, since directions are always drawn in 3D whatever the mesh.
Anything that could vary between modes, such as the axis of revolution, is one
value per mode and so is class data that `cls` still reaches.
**Instead of:** Instance methods, which is what this was until today, or
`staticmethod`, which closes off the class-level extension point for nothing.
**Cost:** None measured. Dispatch is resolved when the mode is traced, and the
Path contributes no array leaves, so it is static to `filter_jit` either way.

## The device tables are grouped by leading index, not flat

`Group the Scene and MeshTopology tables by index space`, 2026-08-24

**Why:** `Scene` was flat, but six of its fields already carried a `face_`
prefix, so the namespace existed without the type system enforcing it. The
prefix was the only thing separating the sign-flipped per-cell `normal` from the
raw per-face one. `MeshGeometry` was nested all along and reads fluently.
**Instead of:** Flat fields on the argument that the device side is a buffer
table, not a model. That argument holds for behaviour, and `Scene` still has no
methods; it never argued for flat names.
**Cost:** Three `__init__` calls instead of one, so the jaxtyping hook no longer
sees across the tables. `Scene.__check_init__` takes that check over and is
strictly better: it runs at construction in production too, not only under the
test hook, and equinox skips it when unflattening, so jit pays nothing.

## A simplex decomposition is one type, used for cells and for faces

`Group the Scene and MeshTopology tables by index space`, 2026-08-24

**Why:** `sample_cell_point` and `sample_face_point` were the same function over
the same structure, which nothing named. `apex` is `None` for a face, so the
seed concatenation is a Python branch on the treedef and is absent from the face
jaxpr rather than compiled as a `cond`.
**Instead of:** Two functions, or one padded to a shared simplex count. The
latter would have destroyed the per-shape specialisation that keeps them
separate traces.

## cell_face_valid and face_num_verts are derived, not stored

`Group the Scene and MeshTopology tables by index space`, 2026-08-24

**Why:** `cell_face` is -1 padded, so the mask is exactly `cell_face >= 0`, and
`face_vertices` likewise gives the count. Storing them was two arrays that had
to be kept consistent with their source for no gain.
**Instead of:** Keeping them as fields. `cell_face_signs` stays stored, since
recovering it from `face_cells` costs a gather rather than a comparison.
**Cost:** Each access rebuilds the array, so `geometry` hoists them out of its
slot loops.

## The axisymmetric crossing follows docs/hyperbolic.typ, in its own notation

`Add AxisymmetricPath`, 2026-08-24

**Why:** The derivation is the author's, and verified. Implementing from a
restatement in different letters cost more to check than it saved.
**Instead of:** A rederivation that had swapped the meanings of `a` and `b`, and
carried a sign error in one asymptote direction. It was deleted.

## The parametrisation is anchored at the ray, not at the hyperbola's centre

`Add AxisymmetricPath`, 2026-08-24

**Why:** The centre can sit arbitrarily far from the ray, and the position is
then recovered as a difference of large quantities. Fixing the gauge so the ray
sits at `v = 1` removes that, and a scanned reference shows it turns a
near-axial crossing from ~20% silently wrong into machine precision.
**Instead of:** A frame built once per ray and held fixed, which is what the
derivation suggests and what the first attempt did.
**Cost:** The frame is rebuilt per cell, six flops, rather than stored.

## The swirl is carried in the ray state, never recomputed

`Add AxisymmetricPath`, 2026-08-24

**Why:** It is exact at launch (`L = r |d_z|`) and conserved thereafter, since a
wall leaves `r` alone and mirroring preserves `|tangent|`. Recovering it from
`1 - |tangent|^2` cancels once the closest approach falls below `sqrt(eps)` of
the radius, which reintroduces a cliff the rest of the design removes.
**Instead of:** Deriving it from the tangent each time, which needs no extra
field.

## Root selection is on the sign of dg/dv, not on ordering

`Add AxisymmetricPath`, 2026-08-24

**Why:** A convex trace can leave through the face it entered by, routinely for
cells against the axis. Choosing by ordering returns the crossing back *into*
the cell, which skips it and charges its length to the previous one. This is the
failure the previous implementation hit.
**Instead of:** Taking the first positive root, or rejecting roots near zero,
which silently eats the genuine exit.

## Ring weighting is a host-side geometry.revolve, not a scene.build flag

`Weight a revolved mesh by Pappus`, 2026-08-24

**Why:** Keeps `scene.build` a dumb flattener and the Pappus fact in `mesh/`,
where the host-side geometry already lives.
**Instead of:** A mode flag threaded through the host/device boundary.
**Cost:** The caller can forget it and get Cartesian cells with a revolved path,
so `CLAUDE.md` states the pairing.

## The axis convention is hardcoded to columns 0 and 1

`Weight a revolved mesh by Pappus`, 2026-08-24

**Why:** Nothing in the repo needs the freedom, and a configurable axis has to
be kept consistent between `paths.py` and `mesh/geometry.py`.
**Instead of:** Static fields on the path plus configurable column selection in
the reader, which the notes list as optional.

## Per-ray constants are built by Path.launch, in a per-mode state

`Build ray state through Path.launch`, 2026-08-24

**Why:** Each mode conserves something different along a ray: the straight mode
its speed, the revolved mode its swirl. Building both in one hook means the
direction is reduced in exactly one place, and the azimuthal component the
reduction discards is still available where the swirl needs it.
**Instead of:** Recomputing per call, which is what `LinearPath.crossing` did
with `|tangent|`, and which the swirl cannot survive at all.

## The project runs in float64

`Run in float64`, 2026-08-24

**Why:** Physics, not graphics. The axisymmetric crossing spends float32's seven
digits entirely on a ray passing close to the axis, where the cliff would move
from one ray in 10^9 to roughly one in 5000.
**Cost:** Set as an import side effect in `raytrax/__init__.py`, since it must
happen before any array is made.

## Prose conventions are written into CLAUDE.md

`81edf4b`, 2026-08-23

**Why:** Docstrings and commit messages had drifted into multi-paragraph design
arguments over several sessions. Stating the rule once was not enough; it needed
to be checked into the repo where it is re-read.
**Instead of:** Relying on a general instruction to be concise.

## Path is genuinely abstract

`1ba1ab1`, 2026-08-23

**Why:** The kinematics belong on the path, not in `transport`, and every mode
has to implement all of them or fail at construction.
**Instead of:** Free functions plus a mode enum.

## The jaxtyping hook is installed in conftest, not in the package

`8b41575`, 2026-08-23

**Why:** Shapes become real assertions under test while costing nothing in
production, and the hook has to run before anything imports `raytrax`.
**Instead of:** Installing it in `raytrax/__init__.py`, which would tax every
run.

## A 2D mesh is read as extruded, with the out-of-plane share in |tangent|

`9cb53b7`, 2026-08-23

**Why:** Directions stay genuine 3D draws, so `travel` is the true 3D path
length and `exp(-kappa s)` needs no correction in any mode. The deficit from
unit length carries the component that is not stored, which is exactly what the
axisymmetric mode later needed for the swirl.
**Instead of:** Renormalising the direction in plane and correcting the distance
by a cosine factor, as the original specification described.

## Dead lanes are refilled from the source mid-march

`7f7eb2c`, 2026-08-22

**Why:** A `vmap`ped step costs O(lanes), not O(alive), so a terminated ray does
not free anything. Refilling its lane is the only thing that recovers the waste.
**Instead of:** Tracing a fixed batch to completion and launching another.

## Russian roulette was removed

`abbae58`, 2026-08-22

**Why:** It existed to end faint rays that held the batch open, but killing a
ray does not free its lane. It bought nothing that refilling does not, and cost
a branch and a random draw per step.
**Instead of:** Keeping it alongside refilling.

## The energy scatter stays outside the vmap

`5d35544`, 2026-08-22

**Why:** Inside it, XLA materialises a per-ray copy of the accumulator and
memory use explodes.

## Cells are padded to max_faces with a null half-space

`5d35544`, 2026-08-22

**Why:** Mixed element types then fit rectangular arrays. The padding is a zero
normal, which `crossing` rejects for free, and an EXTERIOR neighbour, so a
degenerate lookup escapes rather than walking somewhere wrong.
**Instead of:** Ragged or segmented layouts, which a GPU march cannot use.

## Faces must be planar, and warped quads are rejected

`5d35544`, 2026-08-22

**Why:** A warped face breaks the half-space cell model and rays leak through
it. Detecting it at build time is cheap and the failure is otherwise silent.
**Instead of:** Splitting warped quads into triangles, which changes the mesh
the user supplied.

## Face orientation comes from canonical winding, not permutation parity

`5d35544`, 2026-08-22

**Why:** Two cells sharing a face wind it oppositely. `canonical_cycle` starts
at the lowest vertex id and walks the lexicographically smaller direction, so
they agree on the key and differ on one bit.
**Instead of:** Permutation parity, which cannot distinguish the two windings of
a quad, since both can sort with the same parity. Hex meshes need this.

## The hard split is host versus device

`5d35544`, 2026-08-22

**Why:** It is what keeps mixed element types manageable. `mesh/` is pure NumPy
and may be ragged; `scene.py` flattens once into padded rectangular arrays;
everything downstream is JAX and rectangular.
**Instead of:** One representation used on both sides.
