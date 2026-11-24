"""Turn spectral sources into gridpoint datasets with an external command.

Also holds the small subprocess runner shared with the offline ESMF
backend.
"""

from __future__ import annotations

import logging
import subprocess
from pathlib import Path

import xarray as xr

logger = logging.getLogger(__name__)


class SpectralTransformError(RuntimeError):
    """Raised when a spectral-to-grid transform command fails."""


# ===================================================================
# Subprocess and external-tool runners
# ===================================================================


def _run_subprocess(cmd: list[str]) -> None:
    """Run *cmd* and raise on non-zero exit.

    Args:
        cmd: Command and arguments.

    Raises:
        RuntimeError: When the command exits with a non-zero code.
    """
    logger.info("Running command: %s", " ".join(cmd))
    completed = subprocess.run(cmd, check=False, capture_output=True, text=True)
    if completed.returncode != 0:
        raise RuntimeError(
            f"Command failed (exit {completed.returncode}): "
            f"{' '.join(cmd)}\n"
            f"STDOUT:\n{completed.stdout}\n"
            f"STDERR:\n{completed.stderr}"
        )


# ===================================================================
# Spectral-to-gridpoint materialisation
# ===================================================================


def _materialise_spectral_source(
    source: xr.Dataset | str | Path,
    *,
    transform_command: list[str] | tuple[str, ...] | None,
    workdir: Path,
) -> xr.Dataset:
    """Run an external spectral-to-gridpoint transform.

    Args:
        source: Source dataset or path to the spectral input file.
        transform_command: External command with ``{input}`` and
            ``{output}`` placeholders.
        workdir: Working directory for intermediate files.

    Returns:
        The transformed gridpoint dataset.

    Raises:
        ValueError: When *transform_command* is ``None``.
        SpectralTransformError: When the external command fails.
    """
    if transform_command is None:
        raise ValueError(
            "source_kind='spectral' requires an external transform command, for example a CDO sp2gp invocation."
        )

    if isinstance(source, xr.Dataset):
        spectral_in = workdir / "spectral_input.nc"
        source.to_netcdf(spectral_in)
    else:
        spectral_in = Path(source)

    spectral_out = workdir / "spectral_gridpoint.nc"
    cmd = [arg.format(input=str(spectral_in), output=str(spectral_out)) for arg in transform_command]
    try:
        _run_subprocess(cmd)
    except RuntimeError as exc:  # pragma: no cover
        raise SpectralTransformError(str(exc)) from exc

    return xr.load_dataset(spectral_out)
