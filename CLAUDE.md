# raytrax

Monte-Carlo volumetric radiation transport by ray-marching an unstructured mesh,
on the GPU via JAX.

Module layout and the invariants a change must not violate are in
`ARCHITECTURE.md`. Read it before changing anything in `mesh/`, `scene.py` or
`paths.py`, and add to it when a new invariant appears.

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
  that goes in `DECISIONS.md` or `docs/`, and real invariants go in
  `ARCHITECTURE.md`.
- Docstrings should also document *what* each parameter is, not *how* is it used.
  Attempt to describe the responsibility of each parameter in a single line.
- Comment only what the code cannot say for itself, chiefly framework oddities,
  and say *why* the code deviates from idiom.
- Commit messages are one-liners. Add a body only to record a fact the diff
  cannot show, and keep it to a sentence or two.
- `DECISIONS.md` records why a design was chosen and what it was chosen over.
  `ARCHITECTURE.md` states the invariant; that one states the argument.
- Prefer invariance tests over piles of hardcoded cases; if a specific case is
  tested, say what it is looking for.
