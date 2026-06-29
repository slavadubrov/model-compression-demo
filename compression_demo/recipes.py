"""Recipe snippets and optional quantization runners."""

from __future__ import annotations

import json
import pathlib
import shlex
from textwrap import dedent

from .catalog import ALGORITHMS

EXECUTABLE_QUANTIZATION_ALGORITHMS = ("gptq-w4a16", "rtn-w8a16", "fp8-dynamic")

_OUTPUT_SUFFIXES = {
    "gptq-w4a16": "W4A16",
    "rtn-w8a16": "W8A16",
    "fp8-dynamic": "FP8-Dynamic",
}


def _model_slug(model: str) -> str:
    return model.rstrip("/").split("/")[-1].replace(" ", "-")


def default_output_dir(*, model: str, algorithm_key: str) -> str:
    """Return the conventional output directory for a model and algorithm."""

    suffix = _OUTPUT_SUFFIXES.get(algorithm_key, algorithm_key.replace("_", "-"))
    return f"outputs/{_model_slug(model)}-{suffix}"


def load_calibration_records(
    calibration_file: str,
    *,
    text_column: str = "text",
) -> list[dict[str, str]]:
    """Load representative calibration records from JSONL or plain text."""

    path = pathlib.Path(calibration_file)
    if not path.exists():
        raise FileNotFoundError(path)

    records: list[dict[str, str]] = []
    if path.suffix.lower() == ".jsonl":
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            if not line.strip():
                continue
            payload = json.loads(line)
            if text_column not in payload:
                raise ValueError(
                    f"{path}:{line_number} does not contain text column {text_column!r}"
                )
            text = str(payload[text_column]).strip()
            if text:
                records.append({text_column: text})
    else:
        for text in path.read_text(encoding="utf-8").splitlines():
            text = text.strip()
            if text:
                records.append({text_column: text})

    if not records:
        raise ValueError(f"{path} did not contain any calibration text")
    return records


def calibration_source_label(
    *,
    dataset: str,
    dataset_config_name: str,
    calibration_file: str | None,
    text_column: str,
) -> str:
    """Describe whether calibration uses generic or representative data."""

    if calibration_file:
        return f"representative local file {calibration_file} (text column: {text_column})"
    return f"generic demo dataset {dataset} / {dataset_config_name}"


def _quantize_command(
    *,
    algorithm_key: str,
    model: str,
    output_dir: str,
    dataset: str,
    dataset_config_name: str,
    calibration_file: str | None,
    text_column: str,
    num_calibration_samples: int,
    max_seq_length: int,
) -> str:
    command = [
        "uv",
        "run",
        "python",
        "demo.py",
        "quantize",
        "--algorithm",
        algorithm_key,
        "--model",
        model,
        "--output-dir",
        output_dir,
        "--num-calibration-samples",
        str(num_calibration_samples),
        "--max-seq-length",
        str(max_seq_length),
    ]
    if calibration_file:
        command.extend(["--calibration-file", calibration_file, "--text-column", text_column])
    else:
        command.extend(["--dataset", dataset, "--dataset-config-name", dataset_config_name])
    return " ".join(shlex.quote(part) for part in command)


def recipe_snippet(algorithm_key: str) -> str:
    """Return a copy-pasteable recipe snippet for an algorithm."""

    if algorithm_key not in ALGORITHMS:
        raise ValueError(f"Unknown algorithm: {algorithm_key}")

    snippets = {
        "gptq-w4a16": """
            from llmcompressor import oneshot
            from llmcompressor.modifiers.quantization import GPTQModifier

            recipe = GPTQModifier(
                scheme="W4A16",
                targets="Linear",
                ignore=["lm_head"],
            )

            oneshot(
                model="Qwen/Qwen3-8B",
                dataset="wikitext",
                dataset_config_name="wikitext-2-raw-v1",
                recipe=recipe,
                output_dir="outputs/Qwen3-8B-W4A16",
                max_seq_length=4096,
                num_calibration_samples=256,
            )
        """,
        "rtn-w8a16": """
            from llmcompressor import oneshot
            from llmcompressor.modifiers.quantization import QuantizationModifier

            recipe = QuantizationModifier(
                scheme="W8A16",
                targets="Linear",
                ignore=["lm_head"],
            )

            oneshot(
                model="Qwen/Qwen3-8B",
                recipe=recipe,
                output_dir="outputs/Qwen3-8B-W8A16",
            )
        """,
        "fp8-dynamic": """
            from llmcompressor import oneshot
            from llmcompressor.modifiers.quantization import QuantizationModifier
            from transformers import AutoModelForCausalLM, AutoTokenizer

            model_id = "Qwen/Qwen3-8B"
            model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto")
            tokenizer = AutoTokenizer.from_pretrained(model_id)

            recipe = QuantizationModifier(
                targets="Linear",
                scheme="FP8_DYNAMIC",
                ignore=["lm_head"],
            )

            oneshot(model=model, recipe=recipe)
            model.save_pretrained("outputs/Qwen3-0.6B-FP8-Dynamic", save_compressed=True)
            tokenizer.save_pretrained("outputs/Qwen3-0.6B-FP8-Dynamic")
        """,
    }
    return dedent(snippets[algorithm_key]).strip() + "\n"


def dry_run_quantization_command(
    *,
    algorithm_key: str,
    model: str,
    output_dir: str,
    dataset: str,
    dataset_config_name: str,
    calibration_file: str | None = None,
    text_column: str = "text",
    num_calibration_samples: int,
    max_seq_length: int,
) -> str:
    """Return the intended quantization run configuration."""

    algorithm = ALGORITHMS[algorithm_key]
    calibration = calibration_source_label(
        dataset=dataset,
        dataset_config_name=dataset_config_name,
        calibration_file=calibration_file,
        text_column=text_column,
    )
    command = _quantize_command(
        algorithm_key=algorithm_key,
        model=model,
        output_dir=output_dir,
        dataset=dataset,
        dataset_config_name=dataset_config_name,
        calibration_file=calibration_file,
        text_column=text_column,
        num_calibration_samples=num_calibration_samples,
        max_seq_length=max_seq_length,
    )
    return dedent(
        f"""
        Quantization dry run
        --------------------
        Algorithm: {algorithm.name}
        Package:   {algorithm.package}
        Model:     {model}
        Output:    {output_dir}
        Calibration: {calibration}
        Samples:   {num_calibration_samples}
        Sequence:  {max_seq_length}

        Exact command:
        {command}

        Re-run without --dry-run in an environment with the GPU stack installed.
        """
    ).strip()


def build_vllm_serve_command(
    *,
    algorithm_key: str,
    model_path: str,
    max_model_len: int = 4096,
    tensor_parallel_size: int = 1,
    port: int = 8000,
    enable_prefix_caching: bool = False,
    fp8_kv_cache: bool = False,
) -> str:
    """Return a vLLM serving command matched to the selected algorithm."""

    command = ["vllm", "serve", model_path, "--max-model-len", str(max_model_len)]
    if algorithm_key == "fp8-dynamic":
        command.extend(["--quantization", "fp8"])
    if fp8_kv_cache:
        command.extend(["--kv-cache-dtype", "fp8"])
    if enable_prefix_caching:
        command.append("--enable-prefix-caching")
    if tensor_parallel_size > 1:
        command.extend(["--tensor-parallel-size", str(tensor_parallel_size)])
    if port != 8000:
        command.extend(["--port", str(port)])
    return " ".join(shlex.quote(part) for part in command)


def run_llmcompressor_quantization(
    *,
    algorithm_key: str,
    model: str,
    output_dir: str,
    dataset: str,
    dataset_config_name: str,
    calibration_file: str | None = None,
    text_column: str = "text",
    num_calibration_samples: int,
    max_seq_length: int,
) -> None:
    """Run a supported llm-compressor quantization job.

    Heavy dependencies are imported lazily so the planner and tests work on a
    normal Python installation.
    """

    if algorithm_key == "gptq-w4a16":
        from llmcompressor import oneshot
        from llmcompressor.modifiers.quantization import GPTQModifier

        recipe = GPTQModifier(scheme="W4A16", targets="Linear", ignore=["lm_head"])
        oneshot_kwargs = {
            "model": model,
            "recipe": recipe,
            "output_dir": output_dir,
            "max_seq_length": max_seq_length,
            "num_calibration_samples": num_calibration_samples,
        }
        from datasets import Dataset

        if calibration_file:
            records = load_calibration_records(calibration_file, text_column=text_column)
        else:
            from datasets import load_dataset

            raw_records = load_dataset(
                dataset,
                dataset_config_name,
                split=f"train[:{num_calibration_samples}]",
            )
            records = []
            for row in raw_records:
                text = str(row.get(text_column, "")).strip()
                if text:
                    records.append({text_column: text})
            if not records:
                raise ValueError(
                    f"Dataset {dataset}/{dataset_config_name} did not provide text column "
                    f"{text_column!r}"
                )
        oneshot_kwargs["dataset"] = Dataset.from_list(records)
        oneshot(**oneshot_kwargs)
        return

    if algorithm_key == "rtn-w8a16":
        from llmcompressor import oneshot
        from llmcompressor.modifiers.quantization import QuantizationModifier

        recipe = QuantizationModifier(scheme="W8A16", targets="Linear", ignore=["lm_head"])
        oneshot(model=model, recipe=recipe, output_dir=output_dir)
        return

    if algorithm_key == "fp8-dynamic":
        from llmcompressor import oneshot
        from llmcompressor.modifiers.quantization import QuantizationModifier
        from transformers import AutoModelForCausalLM, AutoTokenizer

        hf_model = AutoModelForCausalLM.from_pretrained(model, device_map="auto")
        tokenizer = AutoTokenizer.from_pretrained(model)
        recipe = QuantizationModifier(targets="Linear", scheme="FP8_DYNAMIC", ignore=["lm_head"])
        oneshot(model=hf_model, recipe=recipe)
        hf_model.save_pretrained(output_dir, save_compressed=True)
        tokenizer.save_pretrained(output_dir)
        return

    raise NotImplementedError(
        f"Direct execution for {algorithm_key} is not wired into this demo. "
        "Use the recipe subcommand for the reference implementation."
    )
