# RayTraX

Radiation Transport (X)ccelerator - Monte-Carlo volumetric radiation transport
for a participating, emitting fluid medium, by ray-marching an unstructured mesh
on the GPU via JAX.

Supports mixed-element unstructured grids: triangles and quads in 2D; tets,
hexahedra, wedges and pyramids in 3D.

A 2D mesh is read either as **planar**, a slice through something extruded out
of plane, or as **axisymmetric**, the meridional half-plane of a solid of
revolution. Either way a 2D run carries genuine 3D directions rather than
radiation confined to the plane. `paths.py` documents how the ray state stores
that.

## Design notes

- [`docs/hyperbolic.typ`](docs/hyperbolic.typ) - how a straight 3D ray projects
  to a hyperbola in a revolved 2D mesh, the crossing that needs, and what the
  geometry has to be reweighted to.
- [`DECISIONS.md`](DECISIONS.md) - why the design is the way it is, and what
  each choice was chosen over.
