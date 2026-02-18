"""Backward-compatible script entrypoint."""

from pathlib import Path
import sys


def _ensure_src_on_path() -> None:
    root = Path(__file__).resolve().parent
    src = root / "src"
    if src.is_dir():
        sys.path.insert(0, str(src))


if __name__ == "__main__":
    _ensure_src_on_path()
    from CryoLithe.cli import main

    main()
