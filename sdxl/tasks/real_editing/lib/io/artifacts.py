"""Save and load inversion / editing artifacts in a standardised format."""

from __future__ import annotations

import json
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

import torch
from PIL import Image

from sdxl.tasks.real_editing.lib.inversion.base import InversionConfig, InversionResult
from sdxl.tasks.real_editing.lib.editing.masked_editor import EditResult
from sdxl.tasks.real_editing.lib.models.base import TextCondition


# ======================================================================
# Inversion artifacts
# ======================================================================

def save_inversion_artifacts(
    run_dir: Path,
    result: InversionResult,
    prompt: str = "",
    negative_prompt: str = "",
    extra_meta: Optional[Dict] = None,
) -> None:
    """Persist an ``InversionResult`` to disk."""
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    if result.original_image is not None:
        result.original_image.save(run_dir / "original.png")

    if result.reconstruction_image is not None:
        result.reconstruction_image.save(run_dir / "reconstruction.png")

    if result.x_t is not None:
        torch.save(result.x_t.cpu(), run_dir / "x_t.pt")

    if result.uncond_embeddings is not None:
        torch.save(result.uncond_embeddings.cpu(), run_dir / "uncond_embeddings.pt")

    # Text condition tensors
    if result.text_condition is not None:
        tc = result.text_condition
        cond_dir = run_dir / "text_condition"
        cond_dir.mkdir(exist_ok=True)
        if tc.prompt_embeds is not None:
            torch.save(tc.prompt_embeds.cpu(), cond_dir / "prompt_embeds.pt")
        if tc.negative_prompt_embeds is not None:
            torch.save(tc.negative_prompt_embeds.cpu(), cond_dir / "negative_prompt_embeds.pt")
        if tc.pooled_prompt_embeds is not None:
            torch.save(tc.pooled_prompt_embeds.cpu(), cond_dir / "pooled_prompt_embeds.pt")
        if tc.negative_pooled_prompt_embeds is not None:
            torch.save(tc.negative_pooled_prompt_embeds.cpu(), cond_dir / "negative_pooled_prompt_embeds.pt")
        if tc.add_time_ids is not None:
            torch.save(tc.add_time_ids.cpu(), cond_dir / "add_time_ids.pt")

    # Metadata
    cfg = result.config
    metadata = {
        "schema_version": result.schema_version,
        "created_at": datetime.utcnow().isoformat(),
        "model_family": result.model_family,
        "model_id": result.model_id,
        "inversion_backend": result.inversion_backend,
        "backend_status": result.backend_status,
        "prompt": prompt,
        "negative_prompt": negative_prompt,
        "steps": cfg.steps if cfg else 50,
        "guidance_scale": cfg.guidance_scale if cfg else 1.0,
        "seed": cfg.seed if cfg else 0,
        "num_gd_steps": cfg.num_gd_steps if cfg else 0,
        "gd_step_size": cfg.gd_step_size if cfg else 0.0,
        "use_ipa": cfg.use_ipa if cfg else False,
        "ipa_scale": cfg.ipa_scale if cfg else 0.0,
        "num_inner_steps": cfg.num_inner_steps if cfg else 0,
    }
    if extra_meta:
        metadata.update(extra_meta)
    (run_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )

    # Config
    if cfg is not None:
        (run_dir / "config.json").write_text(
            json.dumps({
                "steps": cfg.steps,
                "guidance_scale": cfg.guidance_scale,
                "seed": cfg.seed,
                "num_gd_steps": cfg.num_gd_steps,
                "gd_step_size": cfg.gd_step_size,
                "optimization_start": cfg.optimization_start,
                "use_ipa": cfg.use_ipa,
                "ipa_scale": cfg.ipa_scale,
                "ipa_weight_name": cfg.ipa_weight_name,
                "num_inner_steps": cfg.num_inner_steps,
                "early_stop_epsilon": cfg.early_stop_epsilon,
            }, indent=2),
            encoding="utf-8",
        )

    # Source manifest
    if result.source_manifest:
        (run_dir / "source_manifest.json").write_text(
            json.dumps(result.source_manifest, indent=2), encoding="utf-8"
        )

    print(f"[OK] Inversion artifacts saved in: {run_dir}")


def load_inversion_artifacts(run_dir: Path) -> InversionResult:
    """Load a previously saved ``InversionResult`` from disk."""
    run_dir = Path(run_dir)
    metadata = json.loads((run_dir / "metadata.json").read_text(encoding="utf-8"))

    x_t = None
    if (run_dir / "x_t.pt").exists():
        x_t = torch.load(run_dir / "x_t.pt", map_location="cpu")

    # Backward compat: old maskedLORA_editing format
    if x_t is None and (run_dir / "source_latents.pt").exists():
        x_t = torch.load(run_dir / "source_latents.pt", map_location="cpu")

    uncond = None
    if (run_dir / "uncond_embeddings.pt").exists():
        uncond = torch.load(run_dir / "uncond_embeddings.pt", map_location="cpu")

    # Text condition
    text_cond = None
    cond_dir = run_dir / "text_condition"
    if cond_dir.exists():
        pe = _load_optional(cond_dir / "prompt_embeds.pt")
        npe = _load_optional(cond_dir / "negative_prompt_embeds.pt")
        ppe = _load_optional(cond_dir / "pooled_prompt_embeds.pt")
        nppe = _load_optional(cond_dir / "negative_pooled_prompt_embeds.pt")
        ati = _load_optional(cond_dir / "add_time_ids.pt")
        text_cond = TextCondition(
            prompt_embeds=pe,
            negative_prompt_embeds=npe,
            pooled_prompt_embeds=ppe,
            negative_pooled_prompt_embeds=nppe,
            add_time_ids=ati,
        )

    original = None
    if (run_dir / "original.png").exists():
        original = Image.open(run_dir / "original.png").convert("RGB")

    reconstruction = None
    if (run_dir / "reconstruction.png").exists():
        reconstruction = Image.open(run_dir / "reconstruction.png").convert("RGB")

    source_manifest = {}
    if (run_dir / "source_manifest.json").exists():
        source_manifest = json.loads(
            (run_dir / "source_manifest.json").read_text(encoding="utf-8")
        )

    config = None
    if (run_dir / "config.json").exists():
        cfg_raw = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
        config = InversionConfig(**{
            k: v for k, v in cfg_raw.items()
            if k in InversionConfig.__dataclass_fields__
        })

    return InversionResult(
        model_family=metadata.get("model_family", ""),
        model_id=metadata.get("model_id", ""),
        inversion_backend=metadata.get("inversion_backend", ""),
        backend_status=metadata.get("backend_status", ""),
        x_t=x_t,
        original_image=original,
        reconstruction_image=reconstruction,
        uncond_embeddings=uncond,
        text_condition=text_cond,
        config=config,
        source_manifest=source_manifest,
    )


# ======================================================================
# Edit artifacts
# ======================================================================

def save_edit_artifacts(
    run_dir: Path,
    result: EditResult,
    output_name: str = "edited_target_only.png",
    metrics: Optional[Dict] = None,
) -> None:
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    result.edited_image.save(run_dir / output_name)

    if result.composite_image is not None:
        result.composite_image.save(run_dir / f"composite_{output_name}")

    if result.edit_meta:
        result.edit_meta["output_name"] = output_name
        (run_dir / "edit_meta.json").write_text(
            json.dumps(result.edit_meta, indent=2), encoding="utf-8"
        )

    if metrics is not None:
        (run_dir / "metrics.json").write_text(
            json.dumps(metrics, indent=2), encoding="utf-8"
        )

    print(f"[OK] Edit artifacts saved: {run_dir / output_name}")


def _load_optional(path: Path):
    if path.exists():
        return torch.load(path, map_location="cpu")
    return None
