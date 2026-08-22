# RayTraX

Radiation Transport (X)ccelerator — Monte-Carlo volumetric radiation transport
for a participating, emitting fluid medium, by ray-marching an unstructured mesh
on the GPU via JAX.

Supports mixed-element unstructured grids: triangles and quads in 2D; tets,
hexahedra, wedges and pyramids in 3D.

A 2D mesh is read as a **planar** domain: a slice through something extruded out
of plane, so a 2D run carries genuine 3D directions rather than radiation
confined to the plane. `paths.py` documents how the ray state stores that.

## Design notes

- [`docs/axisymmetric.typ`](docs/axisymmetric.typ) — how a straight 3D ray
  projects to a hyperbola in a revolved 2D mesh, and the crossing that needs.
  The reduced ray state it rests on is in; the hyperbolic crossing is not.
