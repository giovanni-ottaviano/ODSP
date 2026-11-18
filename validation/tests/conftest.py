import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:      # `import validation` from the repo root
    sys.path.insert(0, str(ROOT))


@pytest.fixture(scope="session")
def odsp_bin() -> Path:
    path = ROOT / "bin" / "odsp"
    if not path.exists():
        pytest.skip("bin/odsp not built: run make -j10 bin/odsp")
    return path
