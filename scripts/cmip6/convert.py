#!/usr/bin/env python
"""Scripting solution for converting CMIP6 data to healpix.

This version splits the regrid+upload into three stages so that each
source file is regridded independently (short wall-time per array
element), then combined and uploaded per dataset.

Intermediate results are stored as NetCDF on Lustre (one file per
source file per pyramid level) to avoid the metadata overhead of Zarr
on a parallel filesystem.

Pipeline
--------
gather_sources → create_weights → plan_regrid → regrid_file → group_for_upload → combine_and_upload
  (singleton)     (singleton)      (singleton)    (array)       (singleton)         (array)
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Iterable

import requests
from reflow import Param, Result, RunDir, Workflow

import grid_doctor as gd

wf = Workflow("cmip6_healpix")

CMOR_MAP = {
    "fx": None,
    "subhr": "PT1M",
    "subdaily": "PT1M",
    "30min": "PT30M",
    "15min": "PT15M",
    "1hr": "PT1H",
    "hour": "PT1H",
    "hr": "PT1H",
    "3hr": "PT3H",
    "6hr": "PT6H",
    "12hr": "PT12H",
    "day": "P1D",
    "1d": "P1D",
    "daily": "P1D",
    "mon": "P1M",
    "monthly": "P1M",
    "season": "P3M",
    "seasonal": "P3M",
    "yr": "P1Y",
    "year": "P1Y",
    "annual": "P1Y",
    "monClim": "P1M",
    "dayClim": "P1D",
}


def cmor_to_iso8601(freq):
    f = str(freq).strip()
    if not f:
        return None
    key = f.lower()
    if key in CMOR_MAP:
        return CMOR_MAP[key]
    m = re.match(r"^(\d+)\s*([a-zA-Z]+)$", key)
    if m:
        n, u = m.groups()
        if u in ("min", "minute", "m", "minutes"):
            return f"PT{n}M"
        if u in ("s", "sec", "second", "seconds"):
            return f"PT{n}S"
        if u in ("h", "hr", "hour", "hours"):
            return f"PT{n}H"
        if u in ("d", "day", "days"):
            return f"P{n}D"
        if u in ("mo", "mon", "month", "months"):
            return f"P{n}M"
        if u in ("y", "yr", "year", "years"):
            return f"P{n}Y"
    if key.startswith("p") or key.startswith("pt"):
        return freq
    if "mon" in key:
        return "P1M"
    if "day" in key:
        return "P1D"
    if "hr" in key or "hour" in key:
        return "PT1H"
    return None


@dataclass
class Cmip6Config:
    realm: str = "atmos"
    product: tuple[str, ...] = ("cmip", "scenariomip")
    time_frequency: str = "6hr"
    variable: tuple[str, ...] = ("pr", "tas")
    experiment: tuple[str, ...] = ("ssp585", "ssp245", "ssp370", "historical")
    ensemble: str = "r1i1p1f1"
    freva_instance: str = "https://nextgems.dkrz.de/api/freva-nextgen/databrowser"
    timeout: int = 120

    @property
    def path(self) -> str:
        return "healpix/cmip6/{experiment}-{ensemble}/{model}/{frequency}"

    def _metadata_search(self, **params) -> dict:
        res = requests.get(
            f"{self.freva_instance}/metadata-search/freva/file",
            params=params,
            timeout=self.timeout,
        )
        res.raise_for_status()
        return res.json()

    def _data_search(self, **params) -> list[str]:
        res = requests.get(
            f"{self.freva_instance}/data-search/freva/file",
            params=params,
            stream=True,
            timeout=self.timeout,
        )
        res.raise_for_status()
        return sorted(line for line in res.iter_lines(decode_unicode=True) if line)

    @property
    def model(self) -> list[str]:
        payload = self._metadata_search(
            realm=self.realm,
            product=self.product,
            experiment=self.experiment,
            ensemble=self.ensemble,
            time_frequency=self.time_frequency,
            variable=self.variable,
        )
        model = payload.get("facets", {}).get("model", [])
        return sorted([m for m, n in zip(model[::2], model[1::2]) if n > 0])

    def has_variable(self, model: str, experiment: str, variable: str) -> bool:
        payload = self._metadata_search(
            realm=self.realm,
            model=model,
            experiment=experiment,
            variable=variable,
            ensemble=self.ensemble,
            time_frequency=self.time_frequency,
        )
        return payload.get("total_count", 0) > 0

    def files_for_variable(
        self, model: str, experiment: str, variable: str
    ) -> list[str]:
        return self._data_search(
            realm=self.realm,
            model=model,
            experiment=experiment,
            variable=variable,
            ensemble=self.ensemble,
            time_frequency=self.time_frequency,
        )

    def combinations(
        self,
        experiments: Iterable[str] | None = None,
        models: Iterable[str] | None = None,
        require_all_experiments: bool = False,
        debug: bool = False,
    ) -> dict[str, list[str]]:
        experiments = tuple(experiments or self.experiment)
        models = tuple(models or self.model)
        result: dict[str, list[str]] = {}

        for model_name in models:
            per_model: dict[str, list[str]] = {}
            for experiment_name in experiments:
                _files: list[str] = []
                valid = True
                for variable_name in self.variable:
                    exists = self.has_variable(
                        model=model_name,
                        experiment=experiment_name,
                        variable=variable_name,
                    )
                    if not exists:
                        valid = False
                        break
                    files = self.files_for_variable(
                        model=model_name,
                        experiment=experiment_name,
                        variable=variable_name,
                    )
                    if not files:
                        valid = False
                        if debug:
                            print(
                                f"SKIP model={model_name} "
                                f"experiment={experiment_name} "
                                f"variable={variable_name} "
                                f"reason=data-search returned no files"
                            )
                        break
                    _files += files
                if not valid:
                    continue
                key = self.path.format(
                    experiment=experiment_name,
                    ensemble=self.ensemble,
                    model=model_name,
                    frequency=cmor_to_iso8601(self.time_frequency),
                )
                per_model[key] = sorted(_files)

            if require_all_experiments:
                if len(per_model) == len(experiments):
                    result.update(per_model)
            else:
                result.update(per_model)
        return result


# ---------------------------------------------------------------------------
# Step 1: Discover source files
# ---------------------------------------------------------------------------
@wf.job(cpus=2, time="00:05:00", mem="1GB", partition="shared", version="3")
def gather_sources(
    variable: Annotated[list[str], Param(help="Select the variables")] = list(
        Cmip6Config.variable
    ),
    freq: Annotated[
        str, Param(help="Target time frequency", short="-f")
    ] = Cmip6Config.time_frequency,
) -> list[tuple[str, list[str]]]:
    """Gather all files that need working on."""
    cfg = Cmip6Config(variable=variable, time_frequency=freq)
    return list(cfg.combinations(require_all_experiments=False).items())


# ---------------------------------------------------------------------------
# Step 2: Create / cache ESMF weight files
# ---------------------------------------------------------------------------
@wf.job(
    cpus=128,
    time="08:00:00",
    partition="compute",
    mem="0",
)
def create_weights(
    paths: Annotated[list[tuple[str, list[str]]], Result(step="gather_sources")],
    weights_dir: Annotated[
        Path, Param(help="Path to the grid weight directory")
    ] = Path("/work/ks1387/healpix-weights"),
) -> list[tuple[str, str, int]]:
    """Create the weight files."""
    import xarray as xr

    out = []
    for s3_path, source_paths in paths:
        print("Opening ", source_paths[0])
        dset = xr.open_dataset(source_paths[0])
        max_level = gd.resolution_to_healpix_level(gd.get_latlon_resolution(dset))
        weights_dir.mkdir(exist_ok=True, parents=True)
        weight_file = gd.cached_weights(dset, level=max_level, cache_path=weights_dir)
        out.append([s3_path, weight_file, max_level])
    return out


# ---------------------------------------------------------------------------
# Step 3: Explode into per-file work items
# ---------------------------------------------------------------------------
@wf.job(cpus=1, time="00:05:00", mem="1GB", partition="shared")
def plan_regrid(
    sources: Annotated[list[tuple[str, list[str]]], Result(step="gather_sources")],
    weights: Annotated[list[tuple[str, str, int]], Result(step="create_weights")],
    run_dir: RunDir,
) -> list[dict]:
    """Flatten (s3_path, [files]) pairs into one work item per file."""
    weight_lookup: dict[str, tuple[str, int]] = {
        s3_path: (wf, ml) for s3_path, wf, ml in weights
    }

    staging = run_dir / "staging"
    items: list[dict] = []

    for s3_path, source_files in sources:
        w_file, max_level = weight_lookup[s3_path]
        safe_dir = s3_path.replace("/", "__")
        out_dir = str(staging / safe_dir)

        for idx, src in enumerate(sorted(source_files)):
            items.append(
                {
                    "s3_path": s3_path,
                    "source_file": src,
                    "weight_file": w_file,
                    "max_level": max_level,
                    "output_dir": out_dir,
                    "file_index": idx,
                }
            )

    print(f"Planned {len(items)} regrid tasks across {len(sources)} datasets")
    return items


# ---------------------------------------------------------------------------
# Step 4: Regrid one source file → full HEALPix pyramid as NetCDF
# ---------------------------------------------------------------------------
@wf.array_job(
    cpus=32,
    time="02:00:00",
    mem="0",
    partition="compute",
    array_parallelism=16,
)
def regrid_file(
    item: Annotated[dict, Result(step="plan_regrid")],
) -> dict:
    """Regrid a single source file and write every pyramid level to NetCDF.

    Produces one NetCDF per level::

        <output_dir>/file_00042_level_7.nc
        <output_dir>/file_00042_level_6.nc
        ...
        <output_dir>/file_00042_level_0.nc
    """
    import xarray as xr

    src = item["source_file"]
    max_level = item["max_level"]
    weight_file = item["weight_file"]
    out_dir = Path(item["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    file_idx = item["file_index"]

    print(f"Regridding {src}")
    ds = xr.open_dataset(src)

    pyramid = gd.create_healpix_pyramid(
        ds, max_level=max_level, weights_path=weight_file
    )

    for level, level_ds in pyramid.items():
        nc_path = out_dir / f"file_{file_idx:05d}_level_{level}.nc"
        level_ds.load().to_netcdf(nc_path)

    return {
        "s3_path": item["s3_path"],
        "output_dir": str(out_dir),
        "max_level": max_level,
    }


# ---------------------------------------------------------------------------
# Step 5: Group per-file results back by dataset
# ---------------------------------------------------------------------------
@wf.job(cpus=1, time="00:05:00", mem="1GB", partition="shared")
def group_for_upload(
    results: Annotated[list[dict], Result(step="regrid_file")],
) -> list[dict]:
    """Gather per-file outputs and group by target S3 path."""
    groups: dict[str, dict] = {}
    for r in results:
        key = r["s3_path"]
        if key not in groups:
            groups[key] = {
                "output_dir": r["output_dir"],
                "max_level": r["max_level"],
            }

    out = [{"s3_path": k, **v} for k, v in sorted(groups.items())]
    for g in out:
        print(f"  {g['s3_path']}: level 0-{g['max_level']}")
    return out


# ---------------------------------------------------------------------------
# Step 6: Combine per-file NetCDFs and upload as Zarr pyramid to S3
# ---------------------------------------------------------------------------
@wf.array_job(
    cpus=32,
    time="08:00:00",
    mem="0",
    partition="compute",
    array_parallelism=8,
)
def combine_and_upload(
    group: Annotated[dict, Result(step="group_for_upload")],
    s3_bucket: Annotated[str, Param(help="Target S3 bucket")] = "cmip6",
    s3_endpoint: Annotated[
        str, Param(help="S3 endpoint URL")
    ] = "https://s3.eu-dkrz-3.dkrz.cloud",
    s3_credentials_file: Annotated[
        Path, Param(help="Path to S3 credentials JSON")
    ] = Path.home() / ".s3-credentials.json",
) -> None:
    """Open per-file NetCDFs at each level, concatenate, and upload."""
    from glob import glob

    import xarray as xr

    s3_path = group["s3_path"]
    out_dir = group["output_dir"]
    max_level = group["max_level"]

    print(f"Combining and uploading {s3_path}")

    # Reassemble the pyramid: for each level, open all per-file
    # NetCDFs and concatenate along time.
    pyramid: dict[int, xr.Dataset] = {}
    for level in range(max_level + 1):
        nc_files = sorted(glob(f"{out_dir}/*_level_{level}.nc"))
        if not nc_files:
            raise FileNotFoundError(
                f"No staging files for {s3_path} level {level} in {out_dir}"
            )
        pyramid[level] = xr.open_mfdataset(
            nc_files,
            parallel=False,
            combine="by_coords",
        )

    s3_options = gd.get_s3_options(s3_endpoint, s3_credentials_file)
    gd.save_pyramid_to_s3(
        pyramid,
        f"{s3_bucket}/{s3_path}",
        s3_options,
        mode="w",
    )
    print(f"Uploaded {s3_path}")


if __name__ == "__main__":
    wf.cli()
