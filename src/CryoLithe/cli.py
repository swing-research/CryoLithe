"""CLI entry points for CryoLithe."""

from __future__ import annotations

from typing import Optional

import typer

from .reconstruct import build_reconstruction_config, run_reconstruction

app = typer.Typer(
    help="CryoLithe command line interface",
    pretty_exceptions_show_locals=False,
    add_completion=False,
)


@app.command("reconstruct")
def reconstruct(
    config: Optional[str] = typer.Option(None, "--config", help="Path to the yaml file"),
    model_dir: Optional[str] = typer.Option(None, "--model-dir", help="Path to trained model directory"),
    proj_file: Optional[str] = typer.Option(None, "--proj-file", help="Path to projections .mrc/.mrcs file"),
    angle_file: Optional[str] = typer.Option(None, "--angle-file", help="Path to tilt-angle file"),
    save_dir: Optional[str] = typer.Option(None, "--save-dir", help="Directory to write reconstruction"),
    save_name: Optional[str] = typer.Option(None, "--save-name", help="Output volume filename"),
    device: int = typer.Option(0, "--device", help="CUDA device id"),
    downsample_projections: bool = typer.Option(
        False, "--downsample-projections/--no-downsample-projections", help="Downsample projections"
    ),
    downsample_factor: float = typer.Option(0.25, "--downsample-factor", help="Downsampling factor"),
    anti_alias: bool = typer.Option(True, "--anti-alias/--no-anti-alias", help="Use antialias while downsampling"),
    n3: Optional[int] = typer.Option(None, "--n3", help="Volume size along z-axis"),
    patch_scale: Optional[float] = typer.Option(None, "--patch-scale", help="Override model patch scale"),
    batch_size: int = typer.Option(100000, "--batch-size", help="Batch size for point inference"),
    num_workers: int = typer.Option(0, "--num-workers", help="DataLoader workers"),
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
    )
) -> None:
    """Download pretrained models from Hugging Face Hub."""
    from huggingface_hub import snapshot_download

    path = snapshot_download(
        repo_id="Vinith2/CryoLithe",
        local_dir=local_dir,
        local_dir_use_symlinks=False,
    )
    typer.echo(path)


def main() -> None:
    app()


if __name__ == "__main__":
    main()
