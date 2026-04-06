"""CLI entry point for goai-bench.

Installed as ``goai-bench`` via ``console_scripts`` in setup.py.
Can also be invoked directly::

    python -m goai_bench.cli --help
"""

import sys
from pathlib import Path


def main() -> None:
    """Launch the benchmark CLI.

    Ensures ``src/`` is importable when running from the repo root
    without ``pip install -e .``, then delegates to the Click command
    defined in ``scripts/run_benchmark.py``.
    """
    src_dir = Path(__file__).resolve().parent.parent.parent
    scripts_dir = src_dir / "scripts"

    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))
    if str(scripts_dir.parent) not in sys.path:
        sys.path.insert(0, str(scripts_dir.parent))

    from scripts.run_benchmark import main as _run  # noqa: WPS433

    _run()


if __name__ == "__main__":
    main()
