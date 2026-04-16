"""Provenance tracking for external code sources."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List


_SOURCES: List[Dict[str, str]] = []


def log_external_source(entry: Dict[str, str]) -> None:
    """Register an external code source for tracking."""
    _SOURCES.append(entry)


def get_all_sources() -> List[Dict[str, str]]:
    return list(_SOURCES)


def generate_external_sources_md(output_path: Path) -> None:
    """Write EXTERNAL_SOURCES.md from all registered provenance entries."""
    from sdxl.tasks.real_editing.lib.inversion.registry import BACKENDS

    rows = []
    for cls in BACKENDS.values():
        inst = cls()
        rows.append(inst.provenance())

    lines = [
        "# External Sources",
        "",
        "This file documents the provenance of all external code used in",
        "the `real_editing/` pipeline.",
        "",
        "| Component | Status | Source | URL | Commit/Tag | License | Paper |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for r in rows:
        lines.append(
            f"| {r.get('name', '')} "
            f"| {r.get('status', '')} "
            f"| {r.get('source', '')} "
            f"| {r.get('url', '')} "
            f"| {r.get('commit', '')} "
            f"| {r.get('license', '')} "
            f"| {r.get('paper', '')} |"
        )
    lines.append("")
    lines.append("## Runtime Dependencies")
    lines.append("")
    lines.append("| Dependency | Source | License | Notes |")
    lines.append("| --- | --- | --- | --- |")
    lines.append(
        "| IP-Adapter weights | h94/IP-Adapter (HuggingFace) | Apache-2.0 "
        "| Downloaded on first use via `pipe.load_ip_adapter()` |"
    )
    lines.append(
        "| CLIP Vision encoder | h94/IP-Adapter models/image_encoder | Apache-2.0 "
        "| Used by Tight Inversion for image conditioning |"
    )
    lines.append(
        "| LoRANetwork | trainscripts/textsliders/lora.py (this repo) | -- "
        "| Architecture-agnostic LoRA implementation |"
    )
    lines.append("")

    output_path = Path(output_path)
    output_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"[OK] {output_path} written.")
