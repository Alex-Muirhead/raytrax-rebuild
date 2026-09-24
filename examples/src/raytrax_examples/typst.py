"""Compile the figures of an example's `figures.typ` to images for its notebook.

A `figures.typ` defines one function per figure and ends in `#standalone(...)`
from `examples/style.typ`. Compiled with no input it publishes the figure names;
compiled with `figure=<name>` it renders that figure alone on a fitted page.
"""

import json
from pathlib import Path

import typst

# The editable install leaves this file at examples/src/raytrax_examples.
ROOT = Path(__file__).resolve().parents[2]


def names(source: Path) -> list[str]:
    """Give the names of the figures a document defines.

    Parameters
    ----------
    source
        The `figures.typ` of an example.

    Returns
    -------
    names
        The figure names, in the order the document lists them.
    """
    return json.loads(typst.query(source, "<figures>", field="value", one=True, root=ROOT))


def compile_figures(source: Path, ppi: float = 200.0) -> list[Path]:
    """Compile every figure of a document to a PNG, for the notebook to show.

    Parameters
    ----------
    source
        The `figures.typ` of an example.
    ppi
        The resolution of the images.

    Returns
    -------
    paths
        The images written, one per figure, under `figures/` beside `source`.
    """
    out = source.parent / "figures"
    out.mkdir(exist_ok=True)
    written = []
    for name in names(source):
        path = out / f"{name}.png"
        typst.compile(source, output=path, format="png", ppi=ppi, root=ROOT, sys_inputs={"figure": name})
        written.append(path)
    return written
