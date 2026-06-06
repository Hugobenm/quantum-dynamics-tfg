from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np


def load_npz(path: str | Path) -> dict[str, Any]:
    """Load an NPZ file into a plain dictionary."""
    data = np.load(Path(path), allow_pickle=False)
    return {key: data[key] for key in data.files}


def save_state(path: str | Path, psi: np.ndarray, **metadata: Any) -> None:
    """Save a spinor state and metadata."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, psi=psi, **metadata)

