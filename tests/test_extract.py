from pathlib import Path
import sys

import numpy as np
import rasterio
from rasterio.transform import from_origin
from rasterio.warp import transform

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from extract import sigma0_to_features, extract_dataset


def test_features():
    vv = np.array([[0.1, 0.0], [np.nan, 0.01]], np.float32)
    vh = np.array([[0.01, 0.001], [0.01, 0.001]], np.float32)
    result = sigma0_to_features(vv, vh)
    assert result.shape == (2, 2, 3)
    np.testing.assert_allclose(result[0, 0], [-10, -20, 10], atol=1e-5)
    assert np.isnan(result[0, 1]).all()
    assert np.isnan(result[1, 0]).all()


def write_sar(path, transform_affine, vv, vh):
    with rasterio.open(path, 'w', driver='GTiff', width=32, height=32,
                       count=2, dtype='float32', crs='EPSG:32618',
                       transform=transform_affine, nodata=-9999) as ds:
        ds.write(np.full((32, 32), vv, np.float32), 1)
        ds.write(np.full((32, 32), vh, np.float32), 2)
        ds.set_band_description(1, 'Sigma0_VV')
        ds.set_band_description(2, 'Sigma0_VH')


def test_extract_dataset(tmp_path):
    s1a = tmp_path / 'S1A_IW_GRDH_1SDV_20260429T230656_foo_sigma0_TC.tif'
    s1b = tmp_path / 'S1A_IW_GRDH_1SDV_20260511T230656_bar_sigma0_TC.tif'
    tr = from_origin(400000, 4100000, 10, 10)
    write_sar(s1a, tr, 0.1, 0.01)
    write_sar(s1b, from_origin(400010, 4100000, 10, 10), 0.1, 0.01)
    # Reference is an S2 band with its own grid.
    s2 = tmp_path / 'S2_B04.tif'
    with rasterio.open(s2, 'w', driver='GTiff', height=32, width=32,
                       count=1, dtype='float32', crs='EPSG:32618', transform=tr) as dst:
        dst.write(np.full((32, 32), 0.2, np.float32), 1)
    lon, lat = transform('EPSG:32618', 'EPSG:4326', [400155], [4099845])
    points = tmp_path / 'points.csv'
    points.write_text(f'point_id,lon,lat\np0,{lon[0]},{lat[0]}\n')
    out = tmp_path / 'out'
    extract_dataset([s1b, s1a], s2, points, out, patch_size=8,
                    optical_paths=[s2])
    output = np.load(out / 'p0.npz')
    assert output['sar_patch_ts'].shape == (2, 8, 8, 3)
    assert output['sar_pixel_ts'].shape == (2, 3)
    assert output['optical_patch'].shape == (8, 8, 1)
    assert output['dates'].tolist() == ['2026-04-29T23:06:56Z', '2026-05-11T23:06:56Z']
    np.testing.assert_allclose(output['sar_pixel_ts'],
                               np.array([[-10, -20, 10], [-10, -20, 10]]), atol=1e-4)
    np.testing.assert_allclose(output['optical_patch'], 0.2)
