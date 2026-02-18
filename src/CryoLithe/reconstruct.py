"""Reconstruction pipeline helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any


def _resolve_path_value(value: Any, base_dir: Path) -> Any:
    if isinstance(value, str):
        p = Path(value)
        if not p.is_absolute():
            return str((base_dir / p).resolve())
        return value
    if isinstance(value, list):
        return [_resolve_path_value(v, base_dir) for v in value]
    return value


def resolve_config_paths(config: dict[str, Any], config_path: str | Path) -> dict[str, Any]:
    """Resolve relative filesystem paths against the YAML file directory."""
    base_dir = Path(config_path).resolve().parent
    path_keys = {"model_dir", "proj_file", "angle_file", "save_dir"}

    resolved = dict(config)
    for key in path_keys:
        if key in resolved and resolved[key] is not None:
            resolved[key] = _resolve_path_value(resolved[key], base_dir)
    return resolved


def load_config(config_path: str | Path) -> dict[str, Any]:
    import yaml

    with open(config_path, "r", encoding="utf-8") as file:
        config = yaml.safe_load(file) or {}
    return resolve_config_paths(config, config_path)


def load_default_config() -> dict[str, Any]:
    import yaml

    defaults_path = Path(__file__).with_name("defaults.yaml")
    with open(defaults_path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file) or {}


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
    config_path: str | Path | None = None,
    overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build runtime config from defaults.yaml + optional user YAML + optional CLI overrides."""
    config = load_default_config()

    if config_path is not None:
        config.update(load_config(config_path))

    if overrides:
        non_none_overrides = {k: v for k, v in overrides.items() if v is not None}
        config.update(_normalize_override_paths(non_none_overrides))

    return config


def validate_reconstruction_config(config: dict[str, Any]) -> None:
    required = ["model_dir", "proj_file", "angle_file", "save_dir", "save_name", "N3"]
    missing = [key for key in required if config.get(key) is None]
    if missing:
        missing_options = ", ".join(f"--{item.replace('_', '-')}" for item in missing)
        raise ValueError(
            f"Missing required config fields: {', '.join(missing)}. "
            f"Provide them via defaults.yaml, --config, or CLI options ({missing_options})."
        )


def _expand_to_len(name: str, value: Any, length: int) -> list[Any]:
    if isinstance(value, list):
        if len(value) != length:
            raise ValueError(
                f"'{name}' must have length {length} when using a list of projections, got {len(value)}."
            )
        return value
    return [value] * length


def _detect_devices(config_device: Any) -> tuple[int, bool, list[int] | None]:
    import torch

    if isinstance(config_device, int):
        return config_device, False, None

    gpus: list[int] = []
    for i in range(torch.cuda.device_count()):
        try:
            torch.cuda.get_device_properties(i)
            gpus.append(i)
        except AssertionError:
            pass

    if not gpus:
        raise RuntimeError("No CUDA devices are available, but non-integer 'device' was provided.")

    multi_gpu = len(gpus) > 1
    if multi_gpu:
        print("Using multiple GPUs")
    print("Using GPUs:", gpus)
    return gpus[0], multi_gpu, gpus


def _run_single_reconstruction(
    *,
    evaluator: Any,
    device: int,
    multi_gpu: bool,
    gpu_ids: list[int] | None,
    proj_file: str,
    angle_file: str,
    n3: int,
    save_dir: str,
    save_name: str,
    downsample: bool,
    downsample_factor: float,
    anti_alias: bool,
    batch_size: int,
    num_workers: int,
) -> str:
    import mrcfile
    import numpy as np
    import torch

    angles = np.loadtxt(angle_file)

    projection = mrcfile.open(proj_file, permissive=True).data
    projection = projection - np.mean(projection)
    projection = projection / np.std(projection)

    if downsample:
        proj_ds_set = []
        for proj in projection:
            proj_t = torch.tensor(proj, device=device, dtype=torch.float32)
            proj_ds = torch.nn.functional.interpolate(
                proj_t[None, None],
                scale_factor=downsample_factor,
                align_corners=True,
                antialias=anti_alias,
                mode="bicubic",
            ).squeeze()
            proj_ds_set.append(proj_ds.cpu().numpy())
        projection = np.array(proj_ds_set)

    n1 = projection.shape[1]
    n2 = projection.shape[2]

    if n1 > n2:
        pad = (n1 - n2) // 2
        projection = np.pad(projection, ((0, 0), (0, 0), (pad, pad)))
    elif n2 > n1:
        pad = (n2 - n1) // 2
        projection = np.pad(projection, ((0, 0), (pad, pad), (0, 0)))

    if n3 > int(max(n1, n2)):
        print("Changed value of N3 to be same as max(N1,N2)")
        n3 = int(max(n1, n2))

    Path(save_dir).mkdir(parents=True, exist_ok=True)

    if multi_gpu:
        vol = evaluator.reconstruct(
            projection=projection,
            angles=angles,
            N3=n3,
            N3_scale=0.5,
            batch_size=batch_size,
            num_workers=num_workers,
            gpu_ids=gpu_ids,
        )
    else:
        vol = evaluator.reconstruct(
            projection=projection,
            angles=angles,
            N3=n3,
            N3_scale=0.5,
            batch_size=batch_size,
            num_workers=num_workers,
        )

    vol = np.moveaxis(vol, 2, 0)
    if n1 > n2:
        vol = vol[:, :, pad:-pad]
    elif n2 > n1:
        vol = vol[:, pad:-pad]

    save_path = str(Path(save_dir) / save_name)
    out = mrcfile.new(save_path, overwrite=True)
    out.set_data(vol.astype(np.float32))
    out.close()

    return save_path


def run_reconstruction(config: dict[str, Any]) -> str | list[str]:
    from .evaluator import Evaluator

    validate_reconstruction_config(config)

    device, multi_gpu, gpu_ids = _detect_devices(config["device"])

    model_path = config["model_dir"]
    batch_size = config["batch_size"]
    downsample = config["downsample_projections"]
    downsample_factor = config["downsample_factor"]
    anti_alias = config["anti_alias"]
    patch_scale = config.get("patch_scale", None)
    save_dir = config["save_dir"]
    num_workers = config.get("num_workers", 0)
    print("num_workers:", num_workers)

    evaluator = Evaluator(model_path=model_path, device=device, patch_scale=patch_scale)

    proj_files = config["proj_file"]
    if not isinstance(proj_files, list):
        proj_files = [proj_files]

    angle_files = _expand_to_len("angle_file", config["angle_file"], len(proj_files))
    save_names = _expand_to_len("save_name", config["save_name"], len(proj_files))
    n3_values = _expand_to_len("N3", config["N3"], len(proj_files))

    saved_paths: list[str] = []
    for idx, proj_file in enumerate(proj_files):
        if len(proj_files) > 1:
            print(f"Reconstructing volume {idx + 1}/{len(proj_files)}: {proj_file}")

        save_path = _run_single_reconstruction(
            evaluator=evaluator,
            device=device,
            multi_gpu=multi_gpu,
            gpu_ids=gpu_ids,
            proj_file=proj_file,
            angle_file=angle_files[idx],
            n3=n3_values[idx],
            save_dir=save_dir,
            save_name=save_names[idx],
            downsample=downsample,
            downsample_factor=downsample_factor,
            anti_alias=anti_alias,
            batch_size=batch_size,
            num_workers=num_workers,
        )
        saved_paths.append(save_path)

    return saved_paths if len(saved_paths) > 1 else saved_paths[0]
