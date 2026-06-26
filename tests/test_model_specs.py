from __future__ import annotations

import json
import pathlib

from compression_demo.model_specs import architecture_from_hf_config


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
