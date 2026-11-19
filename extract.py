"""Module 2: aligned VV/VH features, geographic patches, and time series.

Loads terrain-corrected TWO-BAND linear Sigma0 GeoTIFFs written by Module 1.
Alignment is virtual/windowed; no giant full-scene arrays are created.
assue: 10 SAR acquisitions, patch size = 128, and four sentinel-2 bands (B02, B03, B04 and B08):
    sar_patch_first	(128, 128, 3)	    SAR patch from the first acquisition
    sar_pixel_ts	(10, 3)	            Pixel-level SAR time series
    sar_patch_ts	(10, 128, 128, 3)	SAR patch at each acquisition date
    optical_patch	(128, 128, 4)	    Corresponding Sentinel-2 optical patch
"""
import argparse
import csv
import logging
import re
from contextlib import ExitStack
from pathlib import Path

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.vrt import WarpedVRT
from rasterio.warp import transform
from rasterio.windows import Window


def timestamp_from_name(path: Path) -> str:
    match = re.search(r'_(20\d{6}T\d{6})_', path.name)
    if not match:
        raise ValueError(f'No Sentinel-1 acquisition timestamp in {path.name}')
    stamp = match.group(1)
    return f'{stamp[:4]}-{stamp[4:6]}-{stamp[6:8]}T{stamp[9:11]}:{stamp[11:13]}:{stamp[13:15]}Z'


def band_indices(ds, vv_index: int | None, vh_index: int | None) -> tuple[int, int]:
    """Prefer band descriptions; never guess whether band 1 is VV or VH."""
    labels = [str(x or '').lower() for x in ds.descriptions]

    def find(pol: str, override: int | None):
        if override is not None:
            if not 1 <= override <= ds.count:
                raise ValueError(f'{pol} band index outside 1..{ds.count}')
            return override
        matches = [i for i, label in enumerate(labels, 1)
                   if re.search(rf'(?:^|[_-]){pol}(?:$|[_-])', label)]
        if len(matches) != 1:
            raise ValueError(
                f'Cannot uniquely identify {pol} from band descriptions '
                f'{ds.descriptions}. Pass --{pol}-band explicitly.'
            )
        return matches[0]

    vv, vh = find('vv', vv_index), find('vh', vh_index)
    if vv == vh:
        raise ValueError('VV and VH band indices must be different')
    return vv, vh


def read_float(ds, band: int, window: Window) -> np.ndarray:
    arr = ds.read(band, window=window, masked=True).astype(np.float32)
    return arr.filled(np.nan)


def sigma0_to_features(vv: np.ndarray, vh: np.ndarray) -> np.ndarray:
    """Three dB bands, NaN where either linear backscatter is invalid."""
    if vv.shape != vh.shape:
        raise ValueError('VV/VH must be on the same grid')
    ok = np.isfinite(vv) & np.isfinite(vh) & (vv > 0) & (vh > 0)
    result = np.full((*vv.shape, 3), np.nan, dtype=np.float32)
    result[ok, 0] = 10.0 * np.log10(vv[ok])
    result[ok, 1] = 10.0 * np.log10(vh[ok])
    result[ok, 2] = result[..., 0][ok] - result[..., 1][ok]
    return result


def read_points(csv_path: Path) -> list[dict]:
    with csv_path.open(newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        if not {'point_id', 'lon', 'lat'}.issubset(reader.fieldnames or []):
            raise ValueError('Points CSV needs point_id,lon,lat columns')
        points = [{'point_id': row['point_id'], 'lon': float(row['lon']),
                   'lat': float(row['lat'])} for row in reader]
    if len({p['point_id'] for p in points}) != len(points):
        raise ValueError('Duplicate point_id values')
    for p in points:
        if not re.fullmatch(r'[A-Za-z0-9_-]+', p['point_id']):
            raise ValueError('point_id may contain only letters, digits, _ and -')
        if not (-180 <= p['lon'] <= 180 and -90 <= p['lat'] <= 90):
            raise ValueError(f'Invalid geographic coordinates for {p["point_id"]}')
    return points


def patch_window(reference, lon: float, lat: float, size: int) -> Window:
    if size < 1:
        raise ValueError('Patch size must be positive')
    x, y = transform('EPSG:4326', reference.crs, [lon], [lat])
    row, col = reference.index(x[0], y[0])
    start_row = row - size // 2
    start_col = col - size // 2
    if (start_row < 0 or start_col < 0 or
            start_row + size > reference.height or
            start_col + size > reference.width):
        raise ValueError('Patch extends beyond reference raster')
    return Window(start_col, start_row, size, size)


def extract_dataset(sar_paths: list[Path], reference_path: Path, points_path: Path,
                    out_dir: Path, patch_size: int = 128,
                    optical_paths: list[Path] | None = None,
                    vv_index: int | None = None,
                    vh_index: int | None = None) -> None:
    if not sar_paths:
        raise ValueError('No processed Sentinel-1 images supplied')
    if not points_path.is_file():
        raise FileNotFoundError(points_path)
    out_dir.mkdir(parents=True, exist_ok=True)
    sar_paths = sorted(sar_paths, key=lambda p: (timestamp_from_name(p), p.name))
    times = np.array([timestamp_from_name(p) for p in sar_paths])
    points = read_points(points_path)
    optical_paths = optical_paths or []

    with ExitStack() as stack:   # 动态打开所有文件
        reference = stack.enter_context(rasterio.open(reference_path))
        if reference.crs is None:
            raise ValueError('Reference raster must have a projected/geographic CRS')
        # The Sentinel-2 B04 10m grid is a useful reference, if available.
        grid = dict(crs=reference.crs, transform=reference.transform,
                    width=reference.width, height=reference.height,
                    resampling=Resampling.bilinear)

        sar_vrts = []
        indices = []
        for path in sar_paths:
            src = stack.enter_context(rasterio.open(path))
            if src.crs is None:
                raise ValueError(f'No CRS: {path}')
            indices.append(band_indices(src, vv_index, vh_index))
            sar_vrts.append(stack.enter_context(WarpedVRT(src, **grid)))

        optical_vrts = []
        for path in optical_paths:
            src = stack.enter_context(rasterio.open(path))
            if src.count != 1 or src.crs is None:
                raise ValueError(f'Optical input must be a georeferenced single-band raster: {path}')
            optical_vrts.append(stack.enter_context(WarpedVRT(src, **grid)))

        for p in points:
            pid = p['point_id']
            try:
                window = patch_window(reference, p['lon'], p['lat'], patch_size)
            except ValueError as exc:
                logging.warning('SKIP %s: %s', pid, exc)
                continue
            center = Window(int(window.col_off + patch_size // 2),
                            int(window.row_off + patch_size // 2), 1, 1)

            patches, pixels, coverage = [], [], []
            for vrt, (vv_band, vh_band) in zip(sar_vrts, indices):
                vv = read_float(vrt, vv_band, window)
                vh = read_float(vrt, vh_band, window)
                features = sigma0_to_features(vv, vh)
                patches.append(features)
                coverage.append(np.isfinite(features[..., 0]).mean())
                # Time series at *exactly* the common reference-grid center pixel
                pixel = sigma0_to_features(
                    read_float(vrt, vv_band, center),
                    read_float(vrt, vh_band, center)
                )[0, 0, :]
                pixels.append(pixel)

            sar_patch_ts = np.stack(patches, axis=0)   # (T, size, size, 3)
            sar_pixel_ts = np.stack(pixels, axis=0)     # (T, 3)
            if not np.isfinite(sar_patch_ts[..., 0]).any():
                logging.warning('SKIP %s: no SAR coverage at any date', pid)
                continue
            if optical_vrts:
                optical_patch = np.stack(
                    [read_float(vrt, 1, window) for vrt in optical_vrts], axis=-1
                )                                        # (size, size, C_s2)
            else:
                optical_patch = np.empty((patch_size, patch_size, 0), dtype=np.float32)

            # Pixel layout comes from the reference, not from each raw SAR scene.
            filename = out_dir / f'{pid}.npz'
            np.savez_compressed(
                filename, sar_patch_ts=sar_patch_ts,
                sar_pixel_ts=sar_pixel_ts,
                sar_patch_first=sar_patch_ts[0],
                optical_patch=optical_patch,
                valid_fraction=np.asarray(coverage, dtype=np.float32),
                dates=times, point_id=pid,
                lon=p['lon'], lat=p['lat'],
                crs=str(reference.crs),
                patch_transform=np.asarray(reference.window_transform(window)[:6]),
            )
            logging.info('Saved %s: SAR=%s pixel=%s optical=%s valid_range=[%.2f,%.2f]',
                         filename, sar_patch_ts.shape, sar_pixel_ts.shape,
                         optical_patch.shape, min(coverage), max(coverage))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sar-dir', type=Path, default=Path('processed'))
    parser.add_argument('--reference', type=Path, required=True,
                        help='Prefer a Sentinel-2 10m band; otherwise use a TC SAR GeoTIFF')
    parser.add_argument('--points', type=Path, required=True)
    parser.add_argument('--out', type=Path, default=Path('outputs'))
    parser.add_argument('--patch-size', type=int, default=128)
    parser.add_argument('--s2', type=Path, nargs='*', default=[],
                        help='Optional prepared S2 single-band GeoTIFFs in desired order')
    parser.add_argument('--vv-band', type=int, default=None,
                        help='1-based band index if GeoTIFF lacks meaningful descriptions')
    parser.add_argument('--vh-band', type=int, default=None)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format='%(levelname)s %(message)s')
    sar_paths = list(args.sar_dir.glob('*_sigma0_TC.tif'))
    extract_dataset(sar_paths, args.reference, args.points, args.out,
                    args.patch_size, args.s2, args.vv_band, args.vh_band)


if __name__ == '__main__':
    main()

import numpy as np

data = np.load("outputs/point_001.npz")

# Show all stored arrays
print(data.files)

# Access individual arrays
sar_patch = data["sar_patch_first"]
sar_ts = data["sar_pixel_ts"]
sar_patch_ts = data["sar_patch_ts"]
s2_patch = data["optical_patch"]

print(sar_patch.shape)     # (128, 128, 3)
print(sar_ts.shape)        # (10, 3)
print(sar_patch_ts.shape)  # (10, 128, 128, 3)
print(s2_patch.shape)      # (128, 128, 4)