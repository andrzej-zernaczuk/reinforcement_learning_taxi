"""Path utilities for consistent repo-root resolution."""

from pathlib import Path


def get_repo_root(start: Path | None = None) -> Path:
    """Resolve repository root by searching for pyproject.toml or .git."""
    current = Path(start) if start is not None else Path.cwd()

    for candidate in [current, *current.parents]:
        if (candidate / "pyproject.toml").exists() or (candidate / ".git").exists():
            return candidate

    return current
