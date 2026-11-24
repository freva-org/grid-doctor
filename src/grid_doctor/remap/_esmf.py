"""Generate HEALPix weight files with ESMF.

Turns the source and target [`PolygonMesh`][grid_doctor._geometry.mesh.PolygonMesh]
objects into ESMF meshes and computes the weights, on one of two paths:

- **In-memory** via ESMPy for moderate-size grids.
- **Offline** via ``ESMF_RegridWeightGen`` under MPI for large conservative
  jobs where parallel overlap calculation is beneficial.
"""

from __future__ import annotations

import importlib
import importlib.metadata
import logging
import os
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import TYPE_CHECKING

import numpy as np
import xarray as xr

if TYPE_CHECKING:
    from esmpy import Mesh

    from ..weights import SourceDescription

from .._geometry.healpix import _ellipsoid_or_config
from .._geometry.mesh import (
    PolygonMesh,
    _corner_mesh_from_arrays,
    _polygons_to_corner_arrays,
    _target_healpix_mesh,
)
from .._geometry.spectral import _run_subprocess
from ..metadata import HealpixGrid, Provenance
from ..metadata import _keys as keys
from ..types import (
    Ellipsoid,
    FloatArray,
    RemapMethod,
    SourceKind,
    SourceUnits,
)

logger = logging.getLogger(__name__)


# ===================================================================
# Data-classes
# ===================================================================


@dataclass(frozen=True, slots=True)
class OfflineWeightConfig:
    """Configuration for offline ESMF weight generation.

    Attributes:
        enabled: Enable offline weight generation with
            `ESMF_RegridWeightGen`.
        nproc: Number of MPI ranks used for the offline command.
        mpirun: MPI launcher executable.
        esmf_regrid_weightgen: Name or path of the offline ESMF
            executable.
        weight_only: Request a simplified weight file that contains
            only `row`, `col`, and `S` when the ESMF build supports it.
        netcdf4: Write NetCDF-4 output.
        keep_intermediates: Keep temporary source and destination mesh
            files for inspection.
        workdir: Optional directory for the intermediate mesh files.
    """

    enabled: bool = False
    nproc: int = 1
    esmf_regrid_weightgen: str = "ESMF_RegridWeightGen"
    weight_only: bool = False
    netcdf4: bool = True
    keep_intermediates: bool = False
    workdir: Path | None = None

    @property
    def mpirun(self) -> str:
        """Check if we are in a slurm env to use srun."""
        if os.getenv("SLURM_JOB_ID") is not None:
            return "srun"
        return "mpirun"


# ===================================================================
# Lazy imports for optional heavy dependencies
# ===================================================================


def _require_esmpy() -> ModuleType:
    """Import and return the ``esmpy`` module.

    Raises:
        ImportError: When ESMPy is not installed.
    """
    try:
        esmpy = importlib.import_module("esmpy")
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "ESMPy is required for in-memory weight generation. "
            "Install ESMPy or enable the offline ESMF path."
            "install it with:\n"
            "  conda install -c conda-forge esmpy esmf"
        ) from exc
    return esmpy


# ===================================================================
# ESMPy mesh construction
# ===================================================================


def _mesh_from_polygon_mesh(
    mesh: PolygonMesh,
    *,
    esmpy_mod: ModuleType,
) -> Mesh:
    """Construct an ESMPy :class:`Mesh` from a compact polygon mesh.

    Args:
        mesh: Source or target polygon mesh.
        esmpy_mod: The imported ``esmpy`` module.

    Returns:
        An ``esmpy.Mesh`` instance ready for use with
        ``esmpy.Regrid``.
    """
    valid_counts = np.sum(mesh.face_nodes >= 0, axis=1)
    element_types = np.where(
        valid_counts == 3,
        int(esmpy_mod.MeshElemType.TRI),
        np.where(
            valid_counts == 4,
            int(esmpy_mod.MeshElemType.QUAD),
            valid_counts,
        ),
    ).astype(np.int32, copy=False)

    element_conn = mesh.face_nodes[mesh.face_nodes >= 0].astype(np.int32, copy=False)

    # Interleaved (lon, lat) element coordinates.
    element_coords = np.empty(2 * mesh.face_count, dtype=np.float64)
    element_coords[0::2] = mesh.face_lon
    element_coords[1::2] = mesh.face_lat

    esmf_mesh = esmpy_mod.Mesh(
        parametric_dim=2,
        spatial_dim=2,
        coord_sys=esmpy_mod.CoordSys.SPH_DEG,
    )

    n_nodes = mesh.node_lon.size
    node_coords = np.empty(2 * n_nodes, dtype=np.float64)
    node_coords[0::2] = mesh.node_lon
    node_coords[1::2] = mesh.node_lat

    esmf_mesh.add_nodes(
        node_count=n_nodes,
        node_ids=np.arange(1, n_nodes + 1, dtype=np.int32),
        node_coords=node_coords,
        node_owners=np.zeros(n_nodes, dtype=np.int32),
    )
    esmf_mesh.add_elements(
        element_count=mesh.face_count,
        element_ids=np.arange(1, mesh.face_count + 1, dtype=np.int32),
        element_types=element_types,
        element_conn=element_conn,
        element_coords=element_coords,
    )
    return esmf_mesh


# ===================================================================
# UGRID file I/O
# ===================================================================


def write_ugrid_mesh_file(
    polygons: PolygonMesh | list[tuple[FloatArray, FloatArray]],
    path: str | Path,
    *,
    mesh_name: str,
) -> Path:
    """Write a polygon mesh as a simple UGRID NetCDF file.

    The output conforms to UGRID-1.0 conventions and is accepted by
    `ESMF_RegridWeightGen --src_type UGRID`.

    Args:
        polygons: Either a
            [`PolygonMesh`][grid_doctor._geometry.mesh.PolygonMesh] or
            a list of ``(lon_deg, lat_deg)`` polygon tuples.
        path: Output file path.
        mesh_name: Name used for the mesh topology variable.

    Returns:
        Resolved output path.
    """
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)

    if isinstance(polygons, PolygonMesh):
        mesh = polygons
    else:
        mesh = _corner_mesh_from_arrays(*_polygons_to_corner_arrays(polygons))

    topo = xr.DataArray(
        0,
        attrs={
            "cf_role": "mesh_topology",
            "topology_dimension": 2,
            "node_coordinates": "node_lon node_lat",
            "face_node_connectivity": f"{mesh_name}_face_nodes",
        },
    )
    ds = xr.Dataset(
        data_vars={
            mesh_name: topo,
            "node_lon": (
                "node",
                mesh.node_lon,
                {
                    "standard_name": "longitude",
                    "units": "degrees_east",
                },
            ),
            "node_lat": (
                "node",
                mesh.node_lat,
                {
                    "standard_name": "latitude",
                    "units": "degrees_north",
                },
            ),
            f"{mesh_name}_face_nodes": (
                ("face", "max_face_nodes"),
                mesh.face_nodes,
                {"start_index": 0},
            ),
            "face_lon": (
                "face",
                mesh.face_lon,
                {
                    "standard_name": "longitude",
                    "units": "degrees_east",
                },
            ),
            "face_lat": (
                "face",
                mesh.face_lat,
                {
                    "standard_name": "latitude",
                    "units": "degrees_north",
                },
            ),
        },
        attrs={"Conventions": "UGRID-1.0"},
    )
    face_node_var = f"{mesh_name}_face_nodes"
    encoding: dict[str, dict[str, object]] = {
        str(var): {"_FillValue": None}
        for var in ds.data_vars
        if str(var) != face_node_var
    }
    encoding[face_node_var] = {
        "dtype": "int32",
        "_FillValue": np.int32(-1),
    }
    ds.to_netcdf(output, encoding=encoding, format="NETCDF3_64BIT")

    return output


def run_esmf_regrid_weightgen(
    *,
    src_mesh: str | Path,
    dst_mesh: str | Path,
    weights_path: str | Path,
    method: RemapMethod,
    config: OfflineWeightConfig,
    ignore_unmapped: bool,
    src_regional: bool,
    dst_regional: bool,
) -> Path:
    """Run ``ESMF_RegridWeightGen`` under MPI.

    Args:
        src_mesh: Source UGRID mesh file.
        dst_mesh: Destination UGRID mesh file.
        weights_path: Output NetCDF weight file.
        method: Weight method (``"nearest"`` or ``"conservative"``).
        config: Offline execution configuration.
        ignore_unmapped: Pass ``--ignore_unmapped`` when *True*.
        src_regional: Pass ``--src_regional`` when *True*.
        dst_regional: Pass ``--dst_regional`` when *True*.

    Returns:
        Path to the generated weight file.
    """
    method_arg = "neareststod" if method == "nearest" else "conserve"
    cmd: list[str] = []

    if config.nproc > 1:
        flag = "-n" if config.mpirun == "srun" else "-np"
        cmd.extend([config.mpirun, flag, str(config.nproc)])

    cmd.extend(
        [
            config.esmf_regrid_weightgen,
            "-s",
            str(src_mesh),
            "-d",
            str(dst_mesh),
            "-w",
            str(weights_path),
            "--src_type",
            "UGRID",
            "--dst_type",
            "UGRID",
            "-m",
            method_arg,
        ]
    )
    if config.weight_only:
        cmd.append("--weight_only")
    if config.netcdf4:
        cmd.append("--netcdf4")
    if ignore_unmapped:
        cmd.append("--ignore_unmapped")
    if src_regional:
        cmd.append("--src_regional")
    if dst_regional:
        cmd.append("--dst_regional")

    _run_subprocess(cmd)
    return Path(weights_path)


# ===================================================================
# Offline decision heuristic
# ===================================================================


def _default_offline_enabled(
    *,
    method: RemapMethod,
    source_cell_count: int,
    target_cell_count: int,
    config: OfflineWeightConfig,
) -> bool:
    """Decide whether to use the offline ESMF path automatically.

    The offline path is chosen when:

    - *config.enabled* is explicitly *True*, **or**
    - the method is ``"conservative"``, ``nproc > 1``, and the grid
      is large (source >= 200 000 cells or target >= 1 000 000 cells).

    Args:
        method: Weight-generation method.
        source_cell_count: Number of source cells.
        target_cell_count: Number of target cells.
        config: Offline configuration.

    Returns:
        *True* when the offline path should be used.
    """
    if config.enabled:
        return True
    if method != "conservative":
        return False
    return bool(
        config.nproc > 1
        and (source_cell_count >= 200_000 or target_cell_count >= 1_000_000)
    )


# ===================================================================
# In-memory weight computation
# ===================================================================


def _compute_weights_in_memory(
    *,
    source_mesh: PolygonMesh,
    target_mesh: PolygonMesh,
    weights_path: Path,
    method: RemapMethod,
    ignore_unmapped: bool,
    large_file: bool,
) -> Path:
    """Compute weights with ESMPy using compact mesh arrays.

    Args:
        source_mesh: Source polygon mesh.
        target_mesh: Target polygon mesh.
        weights_path: Output weight file.
        method: Regridding method.
        ignore_unmapped: Ignore unmapped destination cells when *True*.
        large_file: Enable ESMPy large-file support.

    Returns:
        Path to the generated weight file.
    """
    esmpy_mod = _require_esmpy()
    src_esmf = _mesh_from_polygon_mesh(source_mesh, esmpy_mod=esmpy_mod)
    dst_esmf = _mesh_from_polygon_mesh(target_mesh, esmpy_mod=esmpy_mod)

    src_field = esmpy_mod.Field(src_esmf, name="src", meshloc=esmpy_mod.MeshLoc.ELEMENT)
    dst_field = esmpy_mod.Field(dst_esmf, name="dst", meshloc=esmpy_mod.MeshLoc.ELEMENT)
    src_field.data[...] = 0.0
    dst_field.data[...] = 0.0

    regrid_method = (
        esmpy_mod.RegridMethod.NEAREST_STOD
        if method == "nearest"
        else esmpy_mod.RegridMethod.CONSERVE
    )
    kwargs: dict[str, object] = {
        "filename": str(weights_path),
        "regrid_method": regrid_method,
        "unmapped_action": (
            esmpy_mod.UnmappedAction.IGNORE
            if ignore_unmapped
            else esmpy_mod.UnmappedAction.ERROR
        ),
        "ignore_degenerate": True,
        "large_file": large_file,
    }
    if method == "conservative":
        kwargs["norm_type"] = esmpy_mod.NormType.DSTAREA
        kwargs["line_type"] = esmpy_mod.LineType.GREAT_CIRCLE

    regrid = esmpy_mod.Regrid(src_field, dst_field, **kwargs)
    regrid.destroy()
    return weights_path


# ===================================================================
# Main backend entry point
# ===================================================================


def compute_healpix_weights_backend(
    source: xr.Dataset | str | Path,
    level: int,
    *,
    method: RemapMethod,
    nest: bool,
    source_units: SourceUnits,
    weights_path: str | Path | None,
    grid: xr.Dataset | None = None,
    source_kind: SourceKind = "auto",
    ignore_unmapped: bool | None = None,
    large_file: bool = True,
    offline: OfflineWeightConfig | None = None,
    spectral_transform_command: (list[str] | tuple[str, ...] | None) = None,
    ellipsoid: Ellipsoid | None = None,
    source_description: SourceDescription | None = None,
) -> Path:
    """Compute reusable HEALPix weights.

    This is the main backend entry point.  It resolves the source
    description, builds the HEALPix target mesh, and dispatches to
    either the in-memory ESMPy workflow or the offline
    ``ESMF_RegridWeightGen`` path.

    Args:
        source: Source dataset or source path.
        level: HEALPix refinement level.
        method: Weight method (``"nearest"`` or ``"conservative"``).
        nest: Write ring-order rows to an explicit *weights_path* when
            *False*. ESMF always computes nested weights; ring rows are
            renumbered (see
            [`nested_to_ring`][grid_doctor.weights.nested_to_ring]). Files
            at the default path are always nested.
        source_units: Angular unit convention of the source
            coordinates.
        weights_path: Output weight file path. Defaults to the local
            weights directory in the
            [`WeightsRef`][grid_doctor.weights.WeightsRef] layout (see
            [`local_weights_dir`][grid_doctor.weights.local_weights_dir]),
            in nested order; choose ring order when applying them.
        grid: Optional external geometry dataset.
        source_kind: Explicit source representation or ``"auto"``.
        ignore_unmapped: Whether destination cells without source
            coverage should be ignored.
        large_file: Forwarded to ESMPy for the in-memory path.
        offline: Offline weight-generation configuration.
        spectral_transform_command: External command used when
            ``source_kind="spectral"``.
        ellipsoid: Ellipsoid of the target geometry; ``None`` uses
            ``grid_doctor.config.ellipsoid``.
        source_description: Description of *source* from
            [`describe_source`][grid_doctor.weights.describe_source];
            skips describing the source again.

    Returns:
        Path to the generated weight file.

    Raises:
        ValueError: When *method* is not supported.
        FileNotFoundError: When the offline ESMF executable or MPI
            launcher cannot be found.
    """
    if method not in {"nearest", "conservative"}:
        raise ValueError("Only 'nearest' and 'conservative' are supported.")

    from ..weights import (
        WEIGHTS_VERSION,
        WeightsDescription,
        describe_source,
        local_weights_dir,
    )

    offline_cfg = offline or OfflineWeightConfig()
    source_desc = source_description or describe_source(
        source,
        grid=grid,
        source_kind=source_kind,
        source_units=source_units,
        spectral_transform_command=spectral_transform_command,
        workdir=offline_cfg.workdir,
    )
    target_grid = HealpixGrid.from_nest(level, nest=nest, ellipsoid=_ellipsoid_or_config(ellipsoid))
    desc = WeightsDescription(source_desc, target_grid, method)
    target = Path(weights_path) if weights_path is not None else local_weights_dir() / desc.ref.path
    ring = not nest and weights_path is not None  # the default (store) layout holds nested weights
    target.parent.mkdir(parents=True, exist_ok=True)
    # ring: rows renumbered below
    _, target_mesh = _target_healpix_mesh(HealpixGrid(level=level, ellipsoid=target_grid.ellipsoid))

    use_offline = _default_offline_enabled(
        method=method,
        source_cell_count=source_desc.source_mesh.face_count,
        target_cell_count=target_mesh.face_count,
        config=offline_cfg,
    )
    eff_ignore = (
        source_desc.ignore_unmapped if ignore_unmapped is None else ignore_unmapped
    )

    # Provenance: every part of the weights' identity, so a store can be
    # re-verified from the files alone.
    meta: dict[str, str | int | float | bool] = {
        **Provenance(method=method).to_attrs(),
        keys.WEIGHTS_VERSION_ATTR: WEIGHTS_VERSION,
        keys.SOURCE_HASH_ATTR: source_desc.digest,
        keys.LEVEL_ATTR: level,
        keys.ORDER_ATTR: "ring" if ring else "nested",
        keys.ELLIPSOID_ATTR: target_grid.ellipsoid,  # as spelled in ELLIPSOIDS (the ref is lowercase)
        keys.BACKEND_ATTR: "offline-esmf" if use_offline else "esmpy",
        keys.ESMF_VERSION_ATTR: _esmf_version(use_offline, offline_cfg),
        keys.IGNORE_UNMAPPED_ATTR: int(bool(eff_ignore)),
        **source_desc.metadata,
    }
    # Generate and annotate next to *target*, then rename: readers never
    # see a partial weight file, and the scratch dir is always removed.
    with tempfile.TemporaryDirectory(dir=target.parent, prefix=f".{target.name}.") as scratch:
        raw, annotated = Path(scratch) / "raw.nc", Path(scratch) / "annotated.nc"
        if use_offline:
            _run_offline_esmf(
                source_desc=source_desc,
                target_mesh=target_mesh,
                target_path=raw,
                level=level,
                nest=True,
                method=method,
                config=offline_cfg,
                ignore_unmapped=eff_ignore,
            )
        else:
            _compute_weights_in_memory(
                source_mesh=source_desc.source_mesh,
                target_mesh=target_mesh,
                weights_path=raw,
                method=method,
                ignore_unmapped=eff_ignore,
                large_file=large_file,
            )
        with xr.open_dataset(raw) as weights:
            weights = weights.load()
        rows = weights["row" if "row" in weights else "dst_address"].values
        meta[keys.UNMAPPED_CELLS_ATTR] = target_mesh.face_count - np.unique(rows).size
        if ring:
            weights = _renumber_to_ring(weights, level)
        weights.assign_attrs(meta).to_netcdf(annotated)
        os.replace(annotated, target)
    return target


def _renumber_to_ring(weights: xr.Dataset, level: int) -> xr.Dataset:
    """Return nested-order *weights* with their target cells numbered in ring order.

    Renumbers the (1-based) rows and reorders ESMF's per-target-cell
    arrays (dimension ``n_b``).
    """
    from ..weights import nested_to_ring

    row = "row" if "row" in weights else "dst_address"
    ring_of = nested_to_ring(np.arange(12 * 4**level), level)
    weights[row] = weights[row].copy(data=(ring_of[weights[row].values.astype(np.int64) - 1] + 1))
    nested_of = np.argsort(ring_of)  # ring cell i holds nested cell nested_of[i]
    for name in [n for n, v in weights.data_vars.items() if "n_b" in v.dims]:
        weights[name] = weights[name].isel(n_b=nested_of)
    return weights


def _esmf_version(offline: bool, config: OfflineWeightConfig) -> str:
    """Return the version of the ESMF that computes the weights, or ``"unknown"``."""
    if not offline:  # from the package metadata: importing ESMPy would start ESMF and write logs
        try:
            return importlib.metadata.version("esmpy")
        except importlib.metadata.PackageNotFoundError:
            return "unknown"
    try:
        with tempfile.TemporaryDirectory() as scratch:  # ESMF writes PET*.Log files into its working dir
            out = subprocess.run(
                [config.esmf_regrid_weightgen, "--version"], capture_output=True, text=True, timeout=60, cwd=scratch
            ).stdout
    except (OSError, subprocess.SubprocessError):
        return "unknown"
    match = re.search(r"\d+\.\d+\.\d+\S*", out)
    return match.group() if match else "unknown"


# ===================================================================
# Offline ESMF orchestration (extracted for readability)
# ===================================================================


def _run_offline_esmf(
    *,
    source_desc: SourceDescription,
    target_mesh: PolygonMesh,
    target_path: Path,
    level: int,
    nest: bool,
    method: RemapMethod,
    config: OfflineWeightConfig,
    ignore_unmapped: bool,
) -> Path:
    """Orchestrate the offline ESMF weight-generation workflow.

    This helper validates the external executables, writes temporary
    UGRID mesh files, runs ``ESMF_RegridWeightGen``, and cleans up
    intermediate files.

    Args:
        source_desc: Resolved source description.
        target_mesh: HEALPix target polygon mesh.
        target_path: Output weight file path.
        level: HEALPix refinement level.
        nest: Use nested ordering when *True*.
        method: Weight method.
        config: Offline execution configuration.
        ignore_unmapped: Ignore unmapped destination cells.

    Returns:
        Path to the generated weight file.

    Raises:
        FileNotFoundError: When the ESMF executable or MPI launcher
            is missing.
    """
    if shutil.which(config.esmf_regrid_weightgen) is None:
        raise FileNotFoundError(
            f"Could not find {config.esmf_regrid_weightgen!r} in PATH."
        )
    if config.nproc > 1 and shutil.which(config.mpirun) is None:
        raise FileNotFoundError(
            f"Could not find MPI launcher {config.mpirun!r} in PATH."
        )

    if config.workdir is None:
        workdir = Path(tempfile.mkdtemp(prefix="grid_doctor_esmf_"))
    else:
        workdir = config.workdir
        workdir.mkdir(parents=True, exist_ok=True)

    order_tag = "nest" if nest else "ring"
    src_file = write_ugrid_mesh_file(
        source_desc.source_mesh,
        workdir / "source_mesh.nc",
        mesh_name="source_mesh",
    )
    dst_file = write_ugrid_mesh_file(
        target_mesh,
        workdir / f"healpix_level_{level}_{order_tag}.nc",
        mesh_name="target_mesh",
    )

    generated = run_esmf_regrid_weightgen(
        src_mesh=src_file,
        dst_mesh=dst_file,
        weights_path=target_path,
        method=method,
        config=config,
        ignore_unmapped=ignore_unmapped,
        src_regional=source_desc.ignore_unmapped,
        dst_regional=False,
    )

    if not config.keep_intermediates and config.workdir is None:
        shutil.rmtree(workdir, ignore_errors=True)

    return generated
