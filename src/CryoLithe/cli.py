"""CLI entry points for CryoLithe."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer
from ml_collections import config_dict as cd

from .config import (
    HF_MODEL_REPO_ID,
    HF_SAMPLE_DATA_REPO_ID,
    TRAINING_DATA_PATH,
    SMALL_SUBSET_TOMOS,
    build_training_config,
    build_reconstruction_config,
    pick_preferred_model_dir,
)
from .reconstruct import run_reconstruction
from .utils.areTomoToImod import convert_to_imod
from .train_model_real import train_model_real

app = typer.Typer(
    help="CryoLithe command line interface",
    pretty_exceptions_show_locals=False,
    add_completion=False,
)


@app.command("reconstruct")
def reconstruct(
    config: Optional[str] = typer.Option(None, "--config", help="Path to the yaml file."),
    model_dir: Optional[str] = typer.Option(None, "--model-dir", help="Path to trained model directory."),
    proj_file: Optional[str] = typer.Option(None, "--proj-file", help="Path to projections .mrc/.mrcs file."),
    angle_file: Optional[str] = typer.Option(None, "--angle-file", help="Path to tilt-angle file."),
    save_dir: Optional[str] = typer.Option(None, "--save-dir", help="Directory to write reconstruction."),
    save_name: Optional[str] = typer.Option(None, "--save-name", help="Output volume filename."),
    device: int = typer.Option(0, "--device", help="CUDA device id."),
    downsample_projections: bool = typer.Option(
        False, "--downsample-projections/--no-downsample-projections", help="Whether to downsample the input projections for reconstruction. If true, the projections will be downsampled by the specified factor."
    ),
    patch_scale: Optional[float] = typer.Option(None, "--patch-scale", help="Override model patch scale. The only parameter that will influence the reconstruction quality. Default is 1. Greater than one means that effective field of the patch increases, and lower than one means that it decreases."),
    downsample_factor: float = typer.Option(0.25, "--downsample-factor", help="Factor by which to downsample the input projections if downsample_projections is true."),
    anti_alias: bool = typer.Option(True, "--anti-alias/--no-anti-alias", help="Use antialias while downsampling."),
    n3: Optional[int] = typer.Option(None, "--n3", help="Volume size along z-axis, after downsampling."),
    pixel: bool = typer.Option(False, "--pixel", help="Use cryolithe-pixel (longer, slighlty better quality) if True, and cryolithe (faster) is False."),
    batch_size: int = typer.Option(100000, "--batch-size", help="Batch size for point inference. Reduce if memory issue."),
    num_workers: int = typer.Option(0, "--num-workers", help="DataLoader workers."),
) -> None:
    """Reconstruct volume using a YAML config or direct CLI options."""
    overrides = {
        "model_dir": model_dir,
        "proj_file": proj_file,
        "angle_file": angle_file,
        "save_dir": save_dir,
        "save_name": save_name,
        "device": device,
        "downsample_projections": downsample_projections,
        "downsample_factor": downsample_factor,
        "anti_alias": anti_alias,
        "N3": n3,
        "patch_scale": patch_scale,
        "model_variant": "cryolithe-pixel" if pixel else None,
        "batch_size": batch_size,
        "num_workers": num_workers,
    }

    run_reconstruction(
        build_reconstruction_config(
            config_path=config,
            overrides=overrides,
        )
    )


@app.command("download")
def download(
    local_dir: Optional[str] = typer.Option(
        None,
        "--local-dir",
        help="Directory to download model files. If omitted, files stay in HF cache.",
    ),
    update_user_config: bool = typer.Option(
        True,
        "--update-user-config/--no-update-user-config",
        help="Whether to write anything to ~/.cryolithe.yaml.",
    ),
    override_model_dir: bool = typer.Option(
        True,
        "--override-model-dir/--no-override-model-dir",
        help="If model_dir already exists in ~/.cryolithe.yaml, replace it with the newly downloaded path (default: enabled).",
    ),
) -> None:
    """Download pretrained models from Hugging Face Hub."""
    from huggingface_hub import snapshot_download
    import yaml
    from pathlib import Path

    path = snapshot_download(
        repo_id=HF_MODEL_REPO_ID,
        local_dir=local_dir,
        local_dir_use_symlinks=False,
    )
    resolved_path = str(Path(path).resolve())

    if update_user_config:
        user_cfg_path = Path.home() / ".cryolithe.yaml"
        if user_cfg_path.exists():
            with open(user_cfg_path, "r", encoding="utf-8") as f:
                user_cfg = yaml.safe_load(f) or {}
        else:
            user_cfg = {}

        existing_model_dir = user_cfg.get("model_dir")
        if existing_model_dir and not override_model_dir:
            typer.echo(
                f"Keeping existing model_dir in {user_cfg_path}: {existing_model_dir}. "
                "Use --override-model-dir to replace it."
            )
        else:
            user_cfg["model_dir"] = resolved_path
            with open(user_cfg_path, "w", encoding="utf-8") as f:
                yaml.safe_dump(user_cfg, f, sort_keys=False)
            typer.echo(f"Updated {user_cfg_path} with model_dir: {resolved_path}")

    typer.echo(f"Model saved in: {resolved_path}")


@app.command("download-sample-data")
def download_sample_data(
    local_dir: Optional[str] = typer.Option(
        None,
        "--local-dir",
        help="Directory to download sample data. Defaults to ./cryolithe-sample-data in the current working directory.",
    ),
) -> None:
    """Download sample tilt-series data from Hugging Face dataset repo."""
    from huggingface_hub import snapshot_download
    from pathlib import Path

    target_dir = Path(local_dir) if local_dir is not None else (Path.cwd() / "cryolithe-sample-data")
    if not target_dir.is_absolute():
        target_dir = (Path.cwd() / target_dir).resolve()
    target_dir.mkdir(parents=True, exist_ok=True)

    file_path = snapshot_download(
        repo_id=HF_SAMPLE_DATA_REPO_ID,
        local_dir=str(target_dir),
        repo_type="dataset",
        local_dir_use_symlinks=False,
    )
    typer.echo(f"Sample data saved in: {file_path}")



@app.command("AreTomoToImod")
def AreTomoToImod(
    aln_path: str = typer.Option(None, "--aln-path", help="Path to trained model directory"),
    output_path: Optional[str] = typer.Option(None, "--output-path", help="Path to save the xf file. (default: ./)."),
) -> None:
    if not(output_path):
        output_path = "./"
    convert_to_imod(aln_path, output_path)


@app.command("train-model")
def train_model_command(
    config: str = typer.Option(..., "--config", help="Path to the training YAML file."),
    load_checkpoint: bool = typer.Option(
        False,
        "--load-checkpoint/--no-load-checkpoint",
        help="Resume training from output_dir/checkpoint.pth.",
    ),
    device: Optional[int] = typer.Option(None, "--device", help="CUDA device id. Overrides config file.")
) -> None:
    """Train CryoLithe using a YAML config merged with train_model.yaml defaults."""
    training_config = build_training_config(config)
    if device is not None:
        training_config["device"] = device
    train_model_real(
        configs=cd.ConfigDict(training_config),
        load_checkpoint=load_checkpoint,
        device=training_config.get("device", 0),
    )

@app.command("download-training-data")
def download_training_data(
    local_dir: Optional[str] = typer.Option(
        None,
        "--local-dir",
        help="Directory to download training data. Defaults to ./cryolithe-training-data in the current working directory.",
    ),
    small_subset: bool = typer.Option(
        False,
        "--small-subset/--full-dataset",
        help="Whether to download a small subset of the training data for quick testing (default: False). " \
        "If enabled, only the first 4 tomograms will be downloaded. This will take around 20GB of storage instead" \
        "of 600+GB for the full dataset.",
    ),
) -> None:
    """Download training tilt-series data from Hugging Face dataset repo."""
    from huggingface_hub import snapshot_download
    from pathlib import Path

    target_dir = Path(local_dir) if local_dir is not None else (Path.cwd() / "cryolithe-training-data")
    if not target_dir.is_absolute():
        target_dir = (Path.cwd() / target_dir).resolve()
    target_dir.mkdir(parents=True, exist_ok=True)

    allow_patterns = SMALL_SUBSET_TOMOS if small_subset else None

    file_path = snapshot_download(
        repo_id=TRAINING_DATA_PATH,
        local_dir=str(target_dir),
        repo_type="dataset",
        local_dir_use_symlinks=False,
        allow_patterns=allow_patterns,
    )
    typer.echo(f"Sample data saved in: {file_path}")


def main() -> None:
    app()


if __name__ == "__main__":
    main()
