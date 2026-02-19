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
    if len(sys.argv) > 1 and sys.argv[1] not in {"reconstruct", "-h", "--help"} and sys.argv[1].startswith("-"):
        sys.argv.insert(1, "reconstruct")
    from CryoLithe.cli import main

    main()
