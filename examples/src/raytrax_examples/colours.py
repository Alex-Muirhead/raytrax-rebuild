"""The lilaq colour maps, for the plots and renders that Python draws.

Each map is registered with matplotlib as `lilaq.<name>`, so matplotlib and
pyvista take it by name, such as `cmap="lilaq.vik"`.
"""

import functools
import json

import matplotlib as mpl
import typst
from matplotlib.colors import Colormap, LinearSegmentedColormap, ListedColormap

from raytrax_examples.typst import ROOT

# The distinct colours of a qualitative map, rather than a scale.
QUALITATIVE = frozenset({"petroff6", "petroff8", "petroff10", "okabe-ito"})

# Samples of each continuous map, enough that matplotlib's interpolation between
# them cannot be told from lilaq's.
SAMPLES = 256

# The qualitative map that fills each element type in a mesh figure.
CYCLE = "petroff10"
# The matplotlib names of the colour maps for a field of one sign, and for a
# signed field centred on zero.
SEQUENTIAL = "lilaq.magma"
DIVERGING = "lilaq.vik"

# lilaq comes in through style.typ, so Python and the figures use the same version.
# lilaq draws an array map as a linear gradient through it, so the samples do too.
QUERY = f"""
#import "/style.typ": lq
#let qualitative = ({"".join(f"{json.dumps(name)}, " for name in sorted(QUALITATIVE))})
#metadata(dictionary(lq.color.map).pairs().map(((name, map)) => {{
  let colours = if name in qualitative {{
    map
  }} else {{
    let scale = gradient.linear(..map)
    range({SAMPLES}).map(i => scale.sample(i / {SAMPLES - 1} * 100%))
  }}
  (name, colours.map(colour => colour.to-hex()))
}}).to-dict()) <maps>
""".encode()


@functools.cache
def lilaq_maps() -> dict[str, list[str]]:
    """Give every lilaq colour map as hex colours.

    Returns
    -------
    maps
        The colours of each map by its lilaq name. A qualitative map gives its
        distinct colours, a continuous map `SAMPLES` colours along it.
    """
    return json.loads(typst.query(QUERY, "<maps>", field="value", one=True, root=ROOT))


def colourmap(name: str) -> Colormap:
    """Give one lilaq colour map as a matplotlib colour map.

    Parameters
    ----------
    name
        The lilaq name of the map, such as `"vik"`.

    Returns
    -------
    colourmap
        A listed map for a qualitative map, a continuous map otherwise.
    """
    colours = lilaq_maps()[name]
    if name in QUALITATIVE:
        return ListedColormap(colours, name=f"lilaq.{name}")
    return LinearSegmentedColormap.from_list(f"lilaq.{name}", colours, N=SAMPLES)


def cycle(name: str = CYCLE) -> list[str]:
    """Give the colours of a qualitative lilaq map, in order.

    Parameters
    ----------
    name
        The lilaq name of the map.

    Returns
    -------
    colours
        The hex colours.
    """
    return lilaq_maps()[name]


def register() -> None:
    """Register every lilaq colour map with matplotlib as `lilaq.<name>`, if it is not already."""
    for name in lilaq_maps():
        if f"lilaq.{name}" not in mpl.colormaps:
            mpl.colormaps.register(colourmap(name))


register()
