"""Compile the figures of an example's `figures.typ` to standalone files.

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


def compile_figures(source: Path, formats: tuple[str, ...] = ("png", "svg", "pdf"), ppi: float = 200.0) -> list[Path]:
    """Compile every figure of a document, one file per figure and format.

    Parameters
    ----------
    source
        The `figures.typ` of an example.
    formats
        The output formats, as file suffixes.
    ppi
        The resolution of PNG output.

    Returns
    -------
    paths
        The files written, under `figures/` beside `source`.
    """
    out = source.parent / "figures"
    out.mkdir(exist_ok=True)
    written = []
    for name in names(source):
        for suffix in formats:
            path = out / f"{name}.{suffix}"
            typst.compile(
                source,
                output=path,
                format=suffix,
                ppi=ppi if suffix == "png" else None,
                root=ROOT,
                sys_inputs={"figure": name},
            )
            written.append(path)
    return written
