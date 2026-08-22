# RayTraX

Radiation Transport (X)ccelerator — Monte-Carlo volumetric radiation transport
for a participating, emitting fluid medium, by ray-marching an unstructured mesh
on the GPU via JAX.

Supports mixed-element unstructured grids: triangles and quads in 2D; tets,
hexahedra, wedges and pyramids in 3D.

## Design notes

- [`docs/axisymmetric.typ`](docs/axisymmetric.typ) — how a straight 3D ray
  projects to a hyperbola in a revolved 2D mesh, and the crossing that needs.
  Not implemented yet.
