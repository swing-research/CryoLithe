"""Reconstruction pipeline helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any, List, Optional, Tuple, Union

from .config import resolve_or_download_model_dir, validate_reconstruction_config


def _expand_to_len(name: str, value: Any, length: int) -> list[Any]:
    if isinstance(value, list):
        if len(value) != length:
            raise ValueError(
                f"'{name}' must have length {length} when using a list of projections, got {len(value)}."
            )
        return value
    return [value] * length


def _detect_devices(config_device: Any) -> Tuple[int, bool, Optional[List[int]]]:
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
    gpu_ids: Optional[List[int]],
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
    import os
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

    save_path = os.path.join(save_dir, save_name)
    out = mrcfile.new(save_path, overwrite=True)
    out.set_data(vol.astype(np.float32))
    out.close()

    return save_path


def run_reconstruction(config: dict[str, Any]) -> Union[str, List[str]]:
    from .evaluator import Evaluator

    validate_reconstruction_config(config)
    model_path = resolve_or_download_model_dir(config)
    config["model_dir"] = model_path

    device, multi_gpu, gpu_ids = _detect_devices(config["device"])

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
