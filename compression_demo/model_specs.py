"""Model architecture presets and Hugging Face config parsing."""

from __future__ import annotations

import json
import pathlib
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ModelArchitecture:
    name: str
    params_b: float | None
    layers: int
    hidden_size: int
    kv_head_ratio: float
    source: str
    notes: tuple[str, ...] = ()


MODEL_PRESETS: dict[str, ModelArchitecture] = {
    "qwen3-0.6b": ModelArchitecture(
        name="Qwen3 0.6B",
        params_b=0.6,
        layers=28,
        hidden_size=1024,
        kv_head_ratio=0.50,
        source="built-in approximate preset",
        notes=("Check the exact model config before final capacity planning.",),
    ),
    "qwen3-8b": ModelArchitecture(
        name="Qwen3 8B",
        params_b=8.1907,
        layers=36,
        hidden_size=4096,
        kv_head_ratio=8 / 32,
        source="built-in preset from official Hugging Face config",
        notes=("Causal text-generation checkpoint suitable for the text-only harness.",),
    ),
}


def _first_number(config: dict[str, Any], *keys: str) -> int | None:
    for key in keys:
        value = config.get(key)
        if isinstance(value, int | float) and value > 0:
            return int(value)
    return None


def _architecture_fields(config: dict[str, Any]) -> tuple[dict[str, Any], tuple[str, ...]]:
    text_config = config.get("text_config")
    if isinstance(text_config, dict):
        return text_config, ("Read language-model architecture fields from nested text_config.",)
    return config, ()


def architecture_from_hf_config(path: str) -> ModelArchitecture:
    """Read layer, hidden-size, and KV-head-ratio fields from a local HF config."""

    config_path = pathlib.Path(path)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    architecture_config, notes = _architecture_fields(config)
    layers = _first_number(architecture_config, "num_hidden_layers", "n_layer", "num_layers")
    hidden_size = _first_number(architecture_config, "hidden_size", "n_embd", "d_model")
    attention_heads = _first_number(
        architecture_config, "num_attention_heads", "n_head", "num_heads"
    )
    kv_heads = _first_number(
        architecture_config, "num_key_value_heads", "n_kv_heads", "num_kv_heads"
    )

    missing = [
        name
        for name, value in (
            ("num_hidden_layers", layers),
            ("hidden_size", hidden_size),
            ("num_attention_heads", attention_heads),
        )
        if value is None
    ]
    if missing:
        raise ValueError(f"{config_path} is missing required config fields: {', '.join(missing)}")

    if kv_heads is None:
        kv_head_ratio = 1.0
        notes = (*notes, "No num_key_value_heads field found; assuming full multi-head KV cache.")
    else:
        kv_head_ratio = kv_heads / attention_heads

    return ModelArchitecture(
        name=str(
            config.get("_name_or_path")
            or architecture_config.get("_name_or_path")
            or config.get("model_type")
            or config_path.stem
        ),
        params_b=None,
        layers=layers,
        hidden_size=hidden_size,
        kv_head_ratio=kv_head_ratio,
        source=str(config_path),
        notes=notes,
    )


def generic_architecture(
    *, layers: int, hidden_size: int, kv_head_ratio: float
) -> ModelArchitecture:
    return ModelArchitecture(
        name="Generic architecture",
        params_b=None,
        layers=layers,
        hidden_size=hidden_size,
        kv_head_ratio=kv_head_ratio,
        source="CLI defaults",
        notes=(
            "Using generic 7B-style architecture assumptions; pass --model-preset or --hf-config.",
        ),
    )
