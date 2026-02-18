"""Configuration and model-resolution utilities for CryoLithe."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional, Union

HF_MODEL_REPO_ID = "Vinith2/CryoLithe"
PREFERRED_MODEL_DIRS = ("cryolithe", "cryolithe-pixel")


def _resolve_path_value(value: Any, base_dir: Path) -> Any:
    if isinstance(value, str):
        p = Path(value)
        if not p.is_absolute():
            return str((base_dir / p).resolve())
        return value
    if isinstance(value, list):
        return [_resolve_path_value(v, base_dir) for v in value]
    return value


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
    import yaml

    with open(config_path, "r", encoding="utf-8") as file:
        config = yaml.safe_load(file) or {}
    return resolve_config_paths(config, config_path)


def load_default_config() -> dict[str, Any]:
    import yaml

    defaults_path = Path(__file__).with_name("defaults.yaml")
    with open(defaults_path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file) or {}


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
