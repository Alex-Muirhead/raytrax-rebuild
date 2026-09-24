# Known issues

Bugs found but not yet fixed, with enough detail to pick up cold. Remove an
entry when it is fixed and a test covers it.

## `vtk.Mesh.cell_data` is wrong when a mesh repeats an element type across blocks

Found 2026-08-25, while writing `examples/visualise.py`. Not fixed.

`src/raytrax/io/vtk.py`, `Mesh.cell_data`:

```python
blocks = self.source.cell_data_dict[name]
elements = [element.name for element in self.topology.block_elements]
return np.concatenate([np.asarray(blocks[name]) for name in elements])
```

**The bug.** It loops over *blocks* but looks up by *element type*.
`topology.block_elements` has one entry per input block, whereas meshio's
`cell_data_dict[field]` is keyed by element name and each value is already
merged across every block of that type. So a mesh with N blocks of `triangle`
fetches the whole merged triangle array N times and concatenates all N copies.
The result is longer than `num_cells` and in the wrong order. Nothing raises.

**Confirmed**, on a 4x4 fragmented gmsh mesh: 8 triangle blocks and 8 quad
blocks, 6,991 cells, and `cell_data("gmsh:physical")` returned 55,928 values,
which is exactly 8 times too many.

**It depends on the reader, not only on the mesh.** meshio's `.vtu` reader
merges same-type blocks on read, so a round trip through `vtk.write` comes back
with one block per type and reads correctly. The gmsh `.msh` reader keeps one
block per model entity and does not merge. So the same geometry is safe as
`.vtu` and broken as `.msh`.

**When it triggers.** A `.msh` whose model was built from several entities, for
instance by `occ.fragment`. A multi-block CFD grid is the realistic case.

**Why nothing is broken today.** `cylinder`, `plane` and `slab` in
`examples/src/raytrax_examples/meshes.py` each mesh one entity, so every element type
appears in exactly one block and the loop degenerates to the correct thing.
`slab` gets its mixed elements from recombination inside a single surface
rather than from stitching entities together, which is also what keeps it
conformal.

**Shape of the fix.** Walk the blocks in order, keeping a cursor per element
type, and take the next `len(block)` entries from that type's merged array.
Note that `source.cells` also holds the lower-dimensional boundary blocks that
`read` discards, so the walk has to skip those the same way `read` does rather
than assume `source.cells` lines up with `topology.block_elements`.

A test wants a mesh with two blocks of one type. Building one in memory is not
enough, because writing and re-reading a `.vtu` merges them; the test needs
either a `.msh` fixture or a `meshio.Mesh` constructed directly and handed to a
`Mesh` without a round trip.
