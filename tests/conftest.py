import json
from pathlib import Path

import pytest

from sensitive3d.io.osgb import bridge_available

needs_bridge = pytest.mark.skipif(not bridge_available(), reason="osgb_bridge not built (scripts/build_bridge.sh)")


@pytest.fixture(scope="session")
def synthetic(tmp_path_factory) -> Path:
    """The synthetic street dataset (OSGB when the bridge is available, OBJ, GLB)."""
    from sensitive3d.synthetic import generate

    out = tmp_path_factory.mktemp("synthetic")
    formats = ("osgb", "obj", "glb") if bridge_available() else ("obj", "glb")
    generate(out, formats=formats, log=lambda *a: None)
    return out


@pytest.fixture(scope="session")
def ground_truth(synthetic) -> list[dict]:
    return json.loads((synthetic / "obj" / "ground_truth.json").read_text(encoding="utf-8"))
