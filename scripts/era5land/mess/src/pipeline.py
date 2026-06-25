from pathlib import Path

from regridder import ReducedGaussianToGaussianRegridder

# Method = Literal["nearest", "conservative", "conservative_2nd"]
method = "conservative_2nd"
nlat=2560 # 2560
weights = Path(f"weights/weights_rg_to_f{nlat}_{method}.nc")
weights.parent.mkdir(exist_ok=True)

sample_file_e5 = "/work/bm1159/XCES/xces-work/k204229/MYWORK/data/cmor-era5/test/E5_data/ETsf00_IV_2023-01-01_027.grb"
file_tas_grb_e5 = "/work/bm1159/XCES/xces-work/k204229/MYWORK/data/cmor-era5/test/E5_data/ETsf00_1M_2025_167.grb"
sample_file_el = "/work/bm1159/XCES/xces-work/k204229/MYWORK/data/cmor-era5/test/EL_data/ELsf00_1M_2026_183.grb"
file_tas_grb_el = "/work/bm1159/XCES/xces-work/k204229/MYWORK/data/cmor-era5/test/EL_data/ELsf12_1M_2025_167.grb"
file_tas_grb_el = "/work/bm1159/XCES/xces-work/k204229/MYWORK/data/cmor-era5/test/EL_data/ELsf12_1H_2025-02-01_167.grb"
# file_tas_nc = "/work/bm1159/XCES/xces-work/k204229/MYWORK/data/cmor-era5/test/E5_data/tas_e5.nc"
# file_pr_grb = "/work/bm1159/XCES/xces-work/k204229/MYWORK/data/cmor-era5/test/E5_data/ETsf12_1M_2025_228.grb"

# Build weights once, then reuse them.
rg = ReducedGaussianToGaussianRegridder(
    sample_source_file=sample_file_el,
    weights_file=weights,
    # Option A: use a target template file with 1D lat/lon coordinates
    # target_template_file="target_gaussian_template.nc",
    # Option B: generate regular/full Gaussian F640-like target
    # target_nlat=640,
    # target_nlon=1280,
    # For ERA5-Land-like fallback you could instead use:
    grid_type="era5_land",
    method=method,  # or "conservative_2nd" if your ESMPy exposes CONSERVE_2ND
    reuse_weights=True,
    ignore_unmapped=False,
)

for in_file in [
    file_tas_grb_el,
]:
    out_file = in_file.replace(".grb", f"_remapped_{method}.nc")
    print(f"Remapping {in_file} -> {out_file}")
    rg.remap_file(in_file, out_file, variable="t2m")
