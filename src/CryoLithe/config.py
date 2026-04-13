"""Configuration and model-resolution utilities for CryoLithe."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Optional, Union

HF_MODEL_REPO_ID = "sada-group/CryoLithe"
HF_SAMPLE_DATA_REPO_ID = "sada-group/CryoLithe-sample-tiltseries"
PREFERRED_MODEL_DIRS = ("cryolithe", "cryolithe-pixel")

TRAINING_DATA_PATH = "sada-group/CryoLithe-training-dataset"
SMALL_SUBSET_TOMOS = [
    "empiar-11830/tomo_001/*",
    "empiar-11830/tomo_002/*",
    "empiar-11830/tomo_004/*",
    "empiar-11830/tomo_005/*",
]

def _resolve_path_value(value: Any, base_dir: Path) -> Any:
    if isinstance(value, str):
        p = Path(value)
        if not p.is_absolute():
            return str((base_dir / p).resolve())
        return value
    if isinstance(value, list):
        return [_resolve_path_value(v, base_dir) for v in value]
    return value


def _resolve_nested_path_values(value: Any, base_dir: Path, key: Optional[str] = None) -> Any:
    if isinstance(value, dict):
        return {
            nested_key: _resolve_nested_path_values(nested_value, base_dir, nested_key)
            for nested_key, nested_value in value.items()
        }
    if isinstance(value, list):
        return [_resolve_nested_path_values(item, base_dir, key) for item in value]
    if not isinstance(value, str) or not value:
        return value

    path_like_keys = {"model_dir", "proj_file", "angle_file", "save_dir", "root_dir", "pretrain_path", "output_dir"}
    if key in path_like_keys or (key is not None and (key.endswith("_dir") or key.endswith("_path"))):
        return _resolve_path_value(value, base_dir)
    return value


def _load_yaml(config_path: Union[str, Path]) -> dict[str, Any]:
    import yaml

    with open(config_path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file) or {}


def _deep_update(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_update(merged[key], value)
        else:
            merged[key] = value
    return merged


def resolve_config_paths(config: dict[str, Any], config_path: Union[str, Path]) -> dict[str, Any]:
    """Resolve relative filesystem paths against the YAML file directory."""
    base_dir = Path(config_path).resolve().parent
    path_keys = {"model_dir", "proj_file", "angle_file", "save_dir"}

    resolved = dict(config)
    for key in path_keys:
        if key in resolved and resolved[key] is not None:
            resolved[key] = _resolve_path_value(resolved[key], base_dir)
    return resolved


def load_config(config_path: Union[str, Path]) -> dict[str, Any]:
    config = _load_yaml(config_path)
    return resolve_config_paths(config, config_path)


def load_default_config() -> dict[str, Any]:
    defaults_path = Path(__file__).with_name("defaults.yaml")
    return _load_yaml(defaults_path)


def load_training_default_config() -> dict[str, Any]:
    defaults_path = Path(__file__).with_name("train_model.yaml")
    defaults = _load_yaml(defaults_path)
    return _resolve_nested_path_values(defaults, defaults_path.parent)


def load_training_config(config_path: Union[str, Path]) -> dict[str, Any]:
    resolved_path = Path(config_path).resolve()
    config = _load_yaml(resolved_path)
    return _resolve_nested_path_values(config, os.getcwd())
    # return _resolve_nested_path_values(config, resolved_path.parent)


def _validate_dataset_lists(dataset: dict[str, Any]) -> None:
    required_list_keys = ("vol_paths", "projection_paths", "angle_paths")
    lengths = {key: len(dataset.get(key, [])) for key in required_list_keys}
    if len(set(lengths.values())) > 1:
        raise ValueError(
            "dataset vol_paths, projection_paths, and angle_paths must have the same length. "
            f"Got lengths: {lengths}"
        )

    optional_list_keys = ("projection_paths_odd", "projection_paths_even", "z_lims_list")
    expected = lengths["vol_paths"]
    for key in optional_list_keys:
        values = dataset.get(key)
        if values in (None, []):
            continue
        if len(values) != expected:
            raise ValueError(
                f"dataset {key} must have length {expected} to match vol_paths, or be empty/null."
            )


def _subset_optional_list(dataset: dict[str, Any], key: str, indices: list[int]) -> Any:
    values = dataset.get(key)
    if values in (None, []):
        return None
    return [values[index] for index in indices]


def _build_dataset_split(dataset: dict[str, Any], indices: list[int], cache: bool) -> dict[str, Any]:
    split = {
        key: value
        for key, value in dataset.items()
        if key
        not in {
            "vol_paths",
            "projection_paths",
            "projection_paths_odd",
            "projection_paths_even",
            "angle_paths",
            "z_lims_list",
        }
    }
    split["vol_paths"] = [dataset["vol_paths"][index] for index in indices]
    split["projection_paths"] = [dataset["projection_paths"][index] for index in indices]
    split["angle_paths"] = [dataset["angle_paths"][index] for index in indices]
    split["projection_paths_odd"] = _subset_optional_list(dataset, "projection_paths_odd", indices)
    split["projection_paths_even"] = _subset_optional_list(dataset, "projection_paths_even", indices)
    split["z_lims_list"] = _subset_optional_list(dataset, "z_lims_list", indices)
    split["cache"] = cache
    return split


def _build_split_indices(config: dict[str, Any]) -> tuple[list[int], list[int], list[int]]:
    import random

    dataset = config["dataset"]
    splits = config["splits"]
    total = len(dataset.get("vol_paths", []))
    indices = list(range(total))

    if splits.get("mode") == "explicit":
        explicit = splits.get("explicit", {})
        return (
            list(explicit.get("train", [])),
            list(explicit.get("valid", [])),
            list(explicit.get("test", [])),
        )

    train_ratio = splits.get("train", 0.8)
    valid_ratio = splits.get("valid", 0.1)
    test_ratio = splits.get("test", 0.1)
    ratio_sum = train_ratio + valid_ratio + test_ratio
    if abs(ratio_sum - 1.0) > 1e-6:
        raise ValueError(f"Split ratios must sum to 1.0, got {ratio_sum}")

    if splits.get("shuffle", True):
        rng = random.Random(splits.get("seed", 0))
        rng.shuffle(indices)

    train_end = int(total * train_ratio)
    valid_end = train_end + int(total * valid_ratio)
    return indices[:train_end], indices[train_end:valid_end], indices[valid_end:]


def build_training_config(config_path: Union[str, Path]) -> dict[str, Any]:
    config = _deep_update(load_training_default_config(), load_training_config(config_path))

    dataset = config.get("dataset")
    splits = config.get("splits")
    if dataset is None:
        raise ValueError("Training config requires a dataset section.")
    if splits is None:
        raise ValueError("Training config requires a splits section.")

    _validate_dataset_lists(dataset)
    train_indices, valid_indices, test_indices = _build_split_indices(config)
    cache_config = splits.get("cache", {})

    config["train_dataset"] = _build_dataset_split(dataset, train_indices, cache_config.get("train", True))
    config["valid_dataset"] = _build_dataset_split(dataset, valid_indices, cache_config.get("valid", True))
    config["test_dataset"] = _build_dataset_split(dataset, test_indices, cache_config.get("test", False))
    return config


def load_user_config() -> dict[str, Any]:
    """Load user-level config from ~/.cryolithe.yaml if present."""
    user_config_path = Path.home() / ".cryolithe.yaml"
    if not user_config_path.exists():
        return {}
    return load_config(user_config_path)


def _normalize_override_paths(overrides: dict[str, Any]) -> dict[str, Any]:
    """Resolve CLI-provided relative paths against current working directory."""
    path_keys = {"model_dir", "proj_file", "angle_file", "save_dir"}
    normalized = dict(overrides)
    cwd = Path.cwd()

    for key in path_keys:
        value = normalized.get(key)
        if isinstance(value, str) and value:
            p = Path(value)
            normalized[key] = str((cwd / p).resolve()) if not p.is_absolute() else str(p)

    return normalized


def build_reconstruction_config(
    config_path: Optional[Union[str, Path]] = None,
    overrides: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    """Build runtime config from defaults + user config + optional YAML + optional CLI overrides."""
    config = load_default_config()

    if config_path is not None:
        config.update(load_config(config_path))

    if overrides:
        non_none_overrides = {k: v for k, v in overrides.items() if v is not None}
        config.update(_normalize_override_paths(non_none_overrides))

    return config


def _ordered_model_variants(model_variant: Optional[str]) -> tuple[str, ...]:
    if model_variant == "cryolithe-pixel":
        return ("cryolithe-pixel", "cryolithe")
    if model_variant == "cryolithe":
        return ("cryolithe", "cryolithe-pixel")
    return PREFERRED_MODEL_DIRS


def pick_preferred_model_dir(root_dir: Union[str, Path], model_variant: Optional[str] = None) -> Optional[str]:
    """Pick model folder under root with preference: cryolithe, then cryolithe-pixel."""
    root = Path(root_dir).expanduser().resolve()
    if not root.exists():
        return None

    for name in _ordered_model_variants(model_variant):
        candidate = root / name
        if (candidate / "config.json").exists():
            return str(candidate)

    if (root / "config.json").exists():
        return str(root)

    for child in root.iterdir():
        if child.is_dir() and (child / "config.json").exists():
            return str(child)
    return None


def resolve_or_download_model_dir(config: dict[str, Any]) -> str:
    """Resolve model_dir from config/home config or download from HF as fallback."""
    model_variant = config.get("model_variant")

    configured = config.get("model_dir")

    # Assume user gives the model_dir path directly if provided, without needing to pick subfolders.
    if configured:
        return configured

    user_cfg = load_user_config()
    user_model_dir = user_cfg.get("model_dir")
    if user_model_dir:
        resolved = pick_preferred_model_dir(user_model_dir, model_variant=model_variant)
        if resolved is not None:
            return resolved

    from huggingface_hub import snapshot_download

    snapshot_root = snapshot_download(repo_id=HF_MODEL_REPO_ID, local_dir_use_symlinks=False)
    resolved = pick_preferred_model_dir(snapshot_root, model_variant=model_variant)
    if resolved is not None:
        return resolved
    raise ValueError(
        f"Downloaded model snapshot does not contain expected model folders: {PREFERRED_MODEL_DIRS}"
    )


def validate_reconstruction_config(config: dict[str, Any]) -> None:
    required = ["proj_file", "angle_file", "save_dir", "save_name", "N3"]
    missing = [key for key in required if config.get(key) is None]
    if missing:
        missing_options = ", ".join(f"--{item.replace('_', '-')}" for item in missing)
        raise ValueError(
            f"Missing required config fields: {', '.join(missing)}. "
            f"Provide them via defaults.yaml, --config, or CLI options ({missing_options})."
        )
