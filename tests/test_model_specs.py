from __future__ import annotations

import json
import pathlib

from compression_demo.model_specs import MODEL_PRESETS, architecture_from_hf_config


def test_architecture_from_hf_config_derives_kv_head_ratio(tmp_path: pathlib.Path) -> None:
    config_path = tmp_path / "config.json"
    config_path.write_text(
        json.dumps(
            {
                "_name_or_path": "example/model",
                "num_hidden_layers": 24,
                "hidden_size": 2048,
                "num_attention_heads": 16,
                "num_key_value_heads": 4,
            }
        ),
        encoding="utf-8",
    )

    architecture = architecture_from_hf_config(str(config_path))

    assert architecture.name == "example/model"
    assert architecture.layers == 24
    assert architecture.hidden_size == 2048
    assert architecture.kv_head_ratio == 0.25


def test_architecture_from_hf_config_reads_nested_text_config(tmp_path: pathlib.Path) -> None:
    config_path = tmp_path / "qwen3_5_config.json"
    config_path.write_text(
        json.dumps(
            {
                "_name_or_path": "Qwen/Qwen3.5-9B",
                "model_type": "qwen3_5",
                "text_config": {
                    "num_hidden_layers": 32,
                    "hidden_size": 4096,
                    "num_attention_heads": 16,
                    "num_key_value_heads": 4,
                },
            }
        ),
        encoding="utf-8",
    )

    architecture = architecture_from_hf_config(str(config_path))

    assert architecture.name == "Qwen/Qwen3.5-9B"
    assert architecture.layers == 32
    assert architecture.hidden_size == 4096
    assert architecture.kv_head_ratio == 0.25
    assert "nested text_config" in architecture.notes[0]


def test_qwen3_presets_cover_both_sizes() -> None:
    assert MODEL_PRESETS["qwen3-0.6b"].params_b == 0.6
    assert MODEL_PRESETS["qwen3-0.6b"].layers == 28
    assert MODEL_PRESETS["qwen3-8b"].params_b == 8.1907
    assert MODEL_PRESETS["qwen3-8b"].kv_head_ratio == 0.25
