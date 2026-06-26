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
    "qwen2.5-7b": ModelArchitecture(
        name="Qwen2.5 7B",
        params_b=7.6,
        layers=28,
        hidden_size=3584,
        kv_head_ratio=4 / 28,
        source="built-in approximate preset",
        notes=("Uses grouped-query attention assumptions from the common 7B family.",),
    ),
    "llama3-8b": ModelArchitecture(
        name="Llama 3/3.1 8B",
        params_b=8.0,
        layers=32,
        hidden_size=4096,
        kv_head_ratio=0.25,
        source="built-in approximate preset",
        notes=("Good first pass for Llama 3-class 8B models.",),
    ),
    "mistral-7b": ModelArchitecture(
        name="Mistral 7B",
        params_b=7.2,
        layers=32,
        hidden_size=4096,
        kv_head_ratio=0.25,
        source="built-in approximate preset",
        notes=("Good first pass for Mistral 7B-class grouped-query models.",),
    ),
    "mixtral-8x7b": ModelArchitecture(
        name="Mixtral 8x7B",
        params_b=46.7,
        layers=32,
        hidden_size=4096,
        kv_head_ratio=0.25,
        source="built-in approximate preset",
        notes=(
            "Uses total parameter count for memory sizing; active-parameter latency is different.",
        ),
    ),
}


def _first_number(config: dict[str, Any], *keys: str) -> int | None:
    for key in keys:
        value = config.get(key)
        if isinstance(value, int | float) and value > 0:
            return int(value)
    return None


def architecture_from_hf_config(path: str) -> ModelArchitecture:
    """Read layer, hidden-size, and KV-head-ratio fields from a local HF config."""

    config_path = pathlib.Path(path)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    layers = _first_number(config, "num_hidden_layers", "n_layer", "num_layers")
    hidden_size = _first_number(config, "hidden_size", "n_embd", "d_model")
    attention_heads = _first_number(config, "num_attention_heads", "n_head", "num_heads")
    kv_heads = _first_number(config, "num_key_value_heads", "n_kv_heads", "num_kv_heads")

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
        notes = ("No num_key_value_heads field found; assuming full multi-head KV cache.",)
    else:
        kv_head_ratio = kv_heads / attention_heads
        notes = ()

    return ModelArchitecture(
        name=str(config.get("_name_or_path") or config_path.stem),
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
