"""Make the jaxtyping shape annotations assertions rather than documentation.

The hook must run before anything imports `raytrax`, which is why it lives in
conftest rather than in the package.
"""

from jaxtyping import install_import_hook

install_import_hook("raytrax", "beartype.beartype")
