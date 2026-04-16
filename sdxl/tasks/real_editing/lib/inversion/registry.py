"""Registry of available inversion backends."""

from __future__ import annotations

from typing import Dict, Type

from .base import InversionBackend
from .ddim import DDIMInversion
from .tight_inversion import TightInversion

BACKENDS: Dict[str, Type[InversionBackend]] = {
    "ddim": DDIMInversion,
    "tight_inversion": TightInversion,  # primary — DDIM + GD + IP-Adapter
}


def get_backend(name: str) -> InversionBackend:
    """Instantiate and return an inversion backend by name."""
    cls = BACKENDS.get(name)
    if cls is None:
        available = ", ".join(sorted(BACKENDS))
        raise ValueError(
            f"Unknown inversion backend {name!r}. Available: {available}"
        )
    return cls()


def list_backends() -> Dict[str, str]:
    """Return {name: status} for every registered backend."""
    return {name: cls.status for name, cls in BACKENDS.items()}
