# Decisions

Why the design is the way it is, and what each choice was chosen over. Newest
first.

`.claude/ARCHITECTURE.md` states the invariants a reader must not violate. This
file states the argument behind them, which the conventions deliberately keep
out of docstrings, comments and commit messages. Keep entries to the four lines
below; anything longer belongs in `docs/`.

## The march's phases carry `jax.named_scope`, so a kernel can be traced back

`uzqmsvou`, 2026-09-17

**Why:** XLA fuses the whole `while` body and keeps no source file or line in
the HLO it emits, so the profiler's kernel table reads `loop_divide_fusion_3`
and nothing says which of our functions is inside it. `jax.named_scope` is the
one annotation that survives fusion: it lands in each operation's `op_name`, so
`crossing`, `advance`, `attenuate`, `interaction`, `refill` and `scatter` come
back out of the optimised HLO and `kernel_profile.py --attribute` rolls the
kernel times up per scope.
**Instead of:** `jax_include_full_tracebacks_in_locations`, which does not put
`source_file` into the metadata `as_text` prints; and reading the fused jaxpr
primitives, which name `mul x8` and point nowhere.
**Cost:** Six annotations in the hot path that exist only for the profiler, and
a scope rollup that splits a kernel's time by how many operations each scope
put in it, which is a share of instructions rather than of cycles.

## Lane sizing is a decorator in its own module, not a branch inside `trace`

`zxluwltv`, 2026-09-16

**Why:** Choosing a width is about the device, and marching is about the mesh;
keeping both in `transport.py` meant every reader of the march stepped over the
memory analysis first. `tuning.py` imports nothing from the package and names
the width by argument, so `autosized("lanes")` decorates anything with a batch
dimension, and `trace = tuning.autosized("lanes")(launch_and_march)` leaves the
march with one lane count and no policy. Clamping that count to the ray count
stays in the march, where it is bookkeeping rather than tuning: a lane past the
budget launches a ray carrying a quantum nobody emitted.
**Instead of:** A `lanes=None` branch inside `trace`, which also silently
narrowed a width the caller had chosen; an overrun on a width someone asked for
is now theirs to see.
**Cost:** `trace` carries the march's signature rather than one of its own, and
the decorator binds arguments through `inspect.signature` on every call to find
the width whether it was passed by name or by position. `lane_capacity` takes
the callable to price instead of the march's arguments.

## The default lane count is the device's width, not the memory ceiling

`zxluwltv`, 2026-09-16

**Why:** A step is `vmap`ped over every lane, so it costs the same whether the
lane is alive or dead, and the march costs `lanes x steps`. Sizing the batch by
memory made `trace` march a million lanes whose mean ray lives 29.6 steps and
whose longest lives 275: 11 percent utilised, and 10.4 s against 2.1 s at
`core_count x 2048 x 4` lanes on the same case. Anything from 8k to 64k lanes
is within 10 percent there, which a count off the hardware lands inside.
**Instead of:** Keeping the memory ceiling as the target, which measures the
wrong quantity; or autotuning the width, which costs a compile per probe.
**Cost:** `THREADS_PER_CORE` is the per-architecture constant the entry below
set out to avoid, but it is wrong by at most 2x on any recent part and the
optimum is a decade wide. The memory analysis leaves the default path, so a
width that does not fit is now found by the `RESOURCE_EXHAUSTED` retry rather
than predicted; `lane_capacity` stays for a caller that wants the ceiling.

## The lane count is measured off the compiled march, not configured per GPU

`zlumtxks`, 2026-09-03

**Why:** Everything a march holds is linear in lanes, so `lanes` defaulting to
the ray count made memory scale with a number the caller thinks of as accuracy:
`examples/reentry.py` ran at one million rays and died at four. XLA already
knows the answer, so `lane_capacity` compiles the march at two probe widths and
differences the memory analysis. On the re-entry case that prices a lane at 349
bytes and predicts the peak to within a percent at a million lanes.
**Instead of:** A per-GPU constant, as `examples/convergence.py` carried, which
is a number nobody can derive and everybody has to maintain; or counting the
bytes by hand, which the XLA temporaries dominate and a kernel change invalidates.
**Cost:** Two extra compiles per estimate, so a sweep hoists it out of the loop.
The pool XLA reports is a share of the whole card rather than what is free on
it, so an estimate that still overruns is halved and marched again.

## What escapes is banked as it leaves, not read off the lane it died in

`zlumtxks`, 2026-09-03

**Why:** An escaping ray deposits nowhere, so its energy used to be recovered at
the end as whatever the dead lanes still carried. Refilling hands that lane to
another ray and overwrites it: on the re-entry case, whose outflow escapes, the
budget closed to 0.62 instead of 1 the moment the batch was capped. Banking it
in `Deposit.escaped` as the ray terminates is what makes the balance independent
of the lane count.
**Instead of:** Making ESCAPE deposit on its exit face like ABSORB, which
closes the balance too but loses the distinction between energy the boundary
took and energy that left through it.

## A faint ray is stopped and its remainder deposited where it stands

`Stop a ray at an extinction floor, depositing its remainder`, 2026-09-03

**Why:** A ray only ends at an absorbing or escaping boundary, so a mirrored or
optically thick problem has no end condition at all: a lane holding 1e-30 of a
quantum keeps stepping and costs exactly as much as a full one. On the 67k-cell
cylinder that tail was the whole run - marching 65536 rays to completion took
8.0 s at any absorption coefficient, because the cost was set by geometry rather
than by optical depth. A floor at 1e-8 of the quantum takes it to 1.6 s at
kappa = 1 and 0.45 s at kappa = 5, and the deposited field moves by 1e-9.
**Instead of:** Russian roulette, which is a stochastic kill needing a weight
boost to stay unbiased, and which buys nothing that refilling does not; or
dropping the remainder as the original raytrax did, which silently destroyed it.
**Cost:** The remainder lands in one cell rather than spread over the cells the
ray would have reached. That displacement is bounded by the floor, so it is a
knob and not a fixed error, and depositing it keeps the balance exact.

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

## The angular momentum is carried in the ray state, never recomputed

`Add AxisymmetricPath`, 2026-08-24

**Why:** It is exact at launch (`r |q_z|`, one product) and conserved thereafter, since a
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
so `.claude/ARCHITECTURE.md` states the pairing.

## The axis convention is hardcoded to columns 0 and 1

`Weight a revolved mesh by Pappus`, 2026-08-24

**Why:** Nothing in the repo needs the freedom, and a configurable axis has to
be kept consistent between `paths.py` and `mesh/geometry.py`.
**Instead of:** Static fields on the path plus configurable column selection in
the reader, which the notes list as optional.

## Per-ray constants are built by Path.launch, in a per-mode state

`Build ray state through Path.launch`, 2026-08-24

**Why:** Each mode conserves something different along a ray: the straight mode
its speed, the revolved mode its angular momentum. Building both in one hook means the
direction is reduced in exactly one place, and the azimuthal component the
reduction discards is still available where the angular momentum needs it.
**Instead of:** Recomputing per call, which is what `LinearPath.crossing` did
with `|tangent|`, and which the angular momentum cannot survive at all.

## The project runs in float64

`Run in float64`, 2026-08-24

**Why:** Physics, not graphics. The output is compared against a measurement,
and float32's seven digits are not enough to stand behind over a long march. The
axisymmetric crossing is only where that shows first: a ray passing close to the
axis spends all seven, moving the cliff from one ray in 10^9 to about one in
5000.
**Cost:** Set as an import side effect in `raytrax/__init__.py`, since it must
happen before any array is made.

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
axisymmetric mode later needed for the angular momentum.
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

## Interactions.kind is static, and apply traces only the kinds present

`yyorxsuz`, 2026-09-02

**Why:** Under `vmap` a `lax.switch` becomes a `select` that evaluates every
branch, so a table with no diffuse group still made every ray draw a Lambertian
direction on every step. On `slab_planar` that branch was 43% of device time by
itself, and its register pressure cost a further 6% in kernels it is not part
of; dropping it took the step from 2610 to 1065 us. Reading `kind` at trace time
needs it static, so a table that changes a kind recompiles - which is cheap, and
compile time fell from 0.78 to 0.32 s because there is less to compile.
**Instead of:** Leaving the branch and demoting its arithmetic to fp32, which
would have made the dead work cheaper rather than absent.

## Simplex spacings are sorted by a comparator network

`otmszrms`, 2026-09-03

**Why:** `jnp.sort` under `vmap` lowers to a cub segmented radix sort over the
whole batch, at 206 registers per thread and 9.4% occupancy. It was 90.7% of
every source sampling, which made one full-batch sample cost 2.0 s, or five
times an entire 400-step march, and `_refill` pays it every few steps. A network
of at most three comparisons returns the same doubles about 2000 times faster,
so `transport.trace` went from 3.6 to 427 thousand rays per second at four times
the lane count.
**Instead of:** Normalised exponentials, which are the same distribution
(docs/simplex.typ) and the same speed, but change the random stream and lose the
telescoping sum to exactly one. The original raytrax already sorted this way; the
rebuild lost it and this restores it.

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
