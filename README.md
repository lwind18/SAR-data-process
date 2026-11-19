# Research-ready Sentinel-1 GRD pipeline

Two modules plus an optional STAC discovery/downloader:

1. `snap_batch.py` + `graphs/s1_grd_sigma0.xml`: local Sentinel-1 IW GRD VV/VH ZIP or `.SAFE` -> **terrain-corrected, linear Sigma0** multiband GeoTIFF per acquisition, using ESA SNAP GPT.
2. `extract.py`: align multiple Sigma0 products to **one reference grid**, convert VV/VH to dB, calculate VV−VH in dB, and extract matched spatial patches and pixel-level time series using CSV latitude/longitude center points.

Designed to be easy to read/edit, not an industry-scale production service.

## Files

```
sar_research_pipeline/
  graphs/s1_grd_sigma0.xml  # SNAP graph, linear Sigma0 output
  snap_batch.py              # Module 1 orchestrator
  extract.py                 # Module 2 extraction and alignment
  requirements.txt
  center_points.csv          # Replace with your own locations
  raw/                       # Place *.zip and/or *.SAFE products here
  processed/                 # SNAP outputs
  outputs/                   # Point-level compressed .npz results
```

## Installation

Install ESA [SNAP](https://step.esa.int/main/download/snap-download/) including the Sentinel-1/Microwave Toolbox. Ensure `gpt -h` works, or supply `--gpt /path/to/gpt`. To check operator availability:

```
gpt Calibration -h
gpt Terrain-Correction -h
```

Create a Python virtual environment and install dependencies:

```
python -m pip install -r requirements.txt
```

## Module 1: Local GRD -> linear VV/VH Sigma0

Copy **complete** Sentinel-1 `S1A_IW_GRDH_1SDV_....zip` archives OR unpacked `.SAFE` folders into `raw/`. SNAP reads officially named ZIPs without extracting. **Do not point SNAP at the measurement TIFFs alone.**

Inspect before running:

```
python snap_batch.py --raw raw --out processed --dry-run
```

Run:

```
python snap_batch.py --raw raw --out processed --pixel-spacing 10
```

SNAP outputs one file per acquisition:

```
processed/S1A_IW_GRDH_1SDV_20260429T230656_..._sigma0_TC.tif
```

Other options:

```
python snap_batch.py --gpt /full/path/to/gpt.exe --dem 'Copernicus 30m Global DEM' --overwrite
```

The graph uses Apply-Orbit-File -> Remove-GRD-Border-Noise -> ThermalNoiseRemoval -> Calibration -> Terrain-Correction -> Write. **No optional speckle filter** in the default research graph; you can insert one in SNAP Graph Builder if needed. Default target pixel spacing: 10 m. **No radiometric terrain flattening**: the output is terrain-corrected Sigma0, not Gamma0 RTC. SNAP downloads orbit/DEM data when necessary; check network connectivity, operator versions, and coverage. For scenes not covered by the selected DEM, choose a suitable DEM and projection.

**Important:** SNAP XML/operator settings depend on the installed SNAP version. The supplied XML follows documented operators but could not be executed without your installed SNAP and actual input imagery. Open the graph in SNAP Graph Builder or inspect `gpt OPERATOR -h` if your version rejects any parameter. The processing graph outputs **linear backscatter**, so Module 2 must be used for dB conversion.

## Module 2: Align to S2 and extract data at centers

Provide a **reference GeoTIFF** for the desired pixel grid. Best option: Sentinel-2 L2A band B04 at 10 m, already a georeferenced GeoTIFF. If you only want SAR, you can use one of the processed Sentinel-1 GeoTIFFs as reference. A center point is given in WGS84 lon/lat (`EPSG:4326`).

Populate `center_points.csv`:

```
point_id,lon,lat
point_001,-76.3000,36.8500
```

Use a Sentinel-2 reference, and optionally include optical bands (`--s2`) in the paired output:

```
python extract.py \
  --sar-dir processed \
  --reference data/S2_B04.tif \
  --s2 data/S2_B02.tif data/S2_B03.tif data/S2_B04.tif data/S2_B08.tif \
  --points center_points.csv \
  --patch-size 128 \
  --out outputs
```

If you have no Sentinel-2 image yet:

```
python extract.py --sar-dir processed \
  --reference 'processed/S1A_..._sigma0_TC.tif' \
  --points center_points.csv --out outputs
```

The script virtually reprojects each SAR raster to the reference grid using Rasterio `WarpedVRT` and **only reads requested windows**. No full-scene resampled copy is necessary. `--s2` takes *prepared* single-band optical images in the order given, each is also warped to the reference grid; they should already be scaled consistently, atmospherically corrected (e.g. L2A surface reflectance), and cloud-masked for research. This code does not perform cloud masking or spectral reflectance scaling.

Each `outputs/<point_id>.npz` file stores:

| Key | Shape | Notes |
| --- | --- | --- |
| `sar_patch_first` | `(128,128,3)` | First date, VV_dB/VH_dB/(VV_dB−VH_dB) |
| `sar_patch_ts` | `(T,128,128,3)` | Same geographic footprint across T dates |
| `sar_pixel_ts` | `(T,3)` | Center pixel across time |
| `optical_patch` | `(128,128,C_s2)` | Optional Sentinel-2 bands; empty last axis if none |
| `valid_fraction` | `(T,)` | Fraction of usable SAR patch pixels, per date |
| `dates` | `(T,)` | ISO UTC times parsed from Sentinel-1 product names |
| `point_id,lon,lat,crs,patch_transform` | scalar / vector | Geospatial metadata |

To load:

```python
import numpy as np
sample = np.load('outputs/point_001.npz')
print(sample['sar_patch_first'].shape)  # (128,128,3)
print(sample['sar_pixel_ts'].shape)     # (T,3)
print(sample['sar_patch_ts'].shape)     # (T,128,128,3)
print(sample['dates'])
```

If a GeoTIFF does not carry usable band descriptions (`Sigma0_VV`, `Sigma0_VH`), explicitly supply `--vv-band 1 --vh-band 2` **only after checking the actual order**. The script will otherwise stop rather than silently swap polarizations.

Invalid/nodata pixels and nonpositive linear Sigma0 become `NaN`. Dates with missing coverage remain in the output with `NaN`; use `valid_fraction`, spatial masks, and appropriate temporal missingness handling when training models.

```

## References

- ESA SNAP Command-Line Tutorial: https://step.esa.int/docs/tutorials/SNAP_CommandLine_Tutorial.pdf
- ESA Microwave Toolbox FAQ (direct ZIP input): https://step.esa.int/main/toolboxes/microwave-toolbox/microwave-toolbox-faq/
- ESA Sentinel-1 time-series tutorial: https://step.esa.int/docs/tutorials/S1TBX%20Time-series%20analysis%20with%20Sentinel-1.pdf
- PySTAC Client: https://pystac-client.readthedocs.io/en/stable/
- Rasterio WarpedVRT: https://rasterio.readthedocs.io/en/stable/api/rasterio.vrt.html
