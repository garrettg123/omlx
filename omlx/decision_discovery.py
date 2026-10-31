# SPDX-License-Identifier: Apache-2.0
"""Lightweight decision-model family discovery without model dependencies."""

import json
from pathlib import Path
from typing import Literal

DecisionKind = Literal["clef", "openjev", "d3", "d1"]

# Clef ships its decision head next to the backbone shards. It is not part of
# the backbone index, so backbone loaders and quantizers must skip it.
CLEF_HEAD_WEIGHTS = "joint_head.safetensors"
CLEF_HEAD_CONFIG = "joint_head_config.json"
_OPENJEV_HELPER = Path("helper/shim.py")
# Decision 3.0 (vllm-sr d3) scores answer codes with an FP32 readout stored
# next to the backbone shards, described by its decision config.
D3_READOUT_WEIGHTS = "readout.safetensors"
D3_DECISION_CONFIG = "decision_config.json"
D3_FORMAT_ID = "d3-code-readout-v1"


def _model_name_hint(model_path: Path) -> str:
    """
    Return the lowercased name used by the directory-name heuristics.

    HF Hub cache snapshots live at ``models--Org--Name/snapshots/<commit>``,
    so their directory name is a commit hash. Use the repo name there,
    otherwise the model directory name.
    """
    if model_path.parent.name == "snapshots":
        decoded = _decode_hf_cache_model_id(model_path.parent.parent)
        if decoded is not None:
            return decoded[1].rsplit("/", 1)[-1].lower()
    return model_path.name.lower()


def _is_d3_decision_config(path: Path) -> bool:
    try:
        with open(path) as f:
            return json.load(f).get("format_id") == D3_FORMAT_ID
    except (OSError, ValueError, AttributeError):
        return False


def decision_kind(model_path: Path, config: dict | None = None) -> DecisionKind | None:
    """Return the decision-model family of a checkpoint, or None.

    Clef is identified by its joint head files and Decision 3.0 by its
    readout and decision config. LiquidAI d1 declares ``D1Model`` on an
    LFM3-VL or LFM3-Omni config. OpenJev is a plain Qwen3.5 checkpoint, so it
    is identified by its helper directory or, for MLX conversions without the
    helper, by the directory name.
    """
    if (model_path / CLEF_HEAD_WEIGHTS).is_file() and (
        model_path / CLEF_HEAD_CONFIG
    ).is_file():
        return "clef"
    if (model_path / D3_READOUT_WEIGHTS).is_file() and _is_d3_decision_config(
        model_path / D3_DECISION_CONFIG
    ):
        return "d3"
    if config is None:
        try:
            with open(model_path / "config.json") as f:
                config = json.load(f)
        except (OSError, ValueError):
            return None
    model_type = config.get("model_type")
    if model_type in {"lfm3_vl", "lfm3_omni"} and (
        "D1Model" in (config.get("architectures") or ())
        or str((config.get("auto_map") or {}).get("AutoModel", "")).endswith("D1Model")
    ):
        return "d1"
    if model_type != "qwen3_5":
        return None
    if (model_path / _OPENJEV_HELPER).is_file() or "openjev" in _model_name_hint(
        model_path
    ):
        return "openjev"
    return None


def _decode_hf_cache_model_id(path: Path) -> tuple[str, str] | None:
    """Decode models--org--repo into (route_safe_id, repo_id)."""
    name = path.name
    if not name.startswith("models--"):
        return None

    encoded = name[len("models--"):]
    if not encoded:
        return None

    parts = encoded.split("--")
    if len(parts) == 1:
        return parts[0], parts[0]

    repo_name = "--".join(parts[1:])
    return f"{parts[0]}--{repo_name}", f"{parts[0]}/{repo_name}"
