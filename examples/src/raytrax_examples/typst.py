"""Compile a figure of an example's `figures.typ` to an image for its notebook.

A `figures.typ` defines one function per figure and ends in `#standalone(...)`
from `examples/style.typ`. Compiled with `figure=<name>` it renders that figure
alone on a fitted page.
"""

from pathlib import Path

import typst

# The editable install leaves this file at examples/src/raytrax_examples.
ROOT = Path(__file__).resolve().parents[2]


def figure(source: Path, name: str, ppi: float = 200.0) -> Path:
    """Compile one figure of a document to a PNG, for the notebook to show.

    Parameters
    ----------
    source
        The `figures.typ` of an example.
    name
        The name of the figure in that document.
    ppi
        The resolution of the image.

    Returns
    -------
    path
        The image written, under `figures/` beside `source`.
    """
    path = source.parent / "figures" / f"{name}.png"
    path.parent.mkdir(exist_ok=True)
    typst.compile(source, output=path, format="png", ppi=ppi, root=ROOT, sys_inputs={"figure": name})
    return path
