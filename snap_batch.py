"""
Module 1: run a SNAP GPT graph over local Sentinel-1 GRD archives/products.
I can download "ESA SNAP desktop" or intall "esa-snappy" package. the former is better for batch processing
https://step.esa.int/main/download/snap-download/?utm_source=chatgpt.com
requirement:
    ESA SNAP Desktop	Yes	Provides SAR processing capabilities
    SNAP Microwave Toolbox (S1TBX)	Yes	Sentinel-1 calibration, noise removal, terrain correction
    SNAP GPT	Yes	Executes SAR processing automatically from Python
outputs: 
    one calibrated, terrain-corrected GeoTIFF for each Sentinel-1 GRD acquisition. 
    Each GeoTIFF contains two bands: VV and VH, stored as linear Sigma0 backscatter values
"""
import argparse
import logging
import shutil
import subprocess
from pathlib import Path


def find_scenes(raw_dir: Path) -> list[Path]:
    """Only accept SAFE products or official Sentinel-1 ZIP archives."""
    scenes = []
    for path in raw_dir.iterdir():
        if path.is_dir() and path.name.upper().endswith('.SAFE'):
            manifest = path / 'manifest.safe'
            if manifest.exists():
                scenes.append(manifest)
            else:
                logging.warning('Skipping incomplete SAFE product: %s', path)
        elif path.is_file() and path.name.upper().startswith('S1') and path.suffix.lower() == '.zip':
            scenes.append(path)
    return sorted(scenes)


def scene_id(scene: Path) -> str:
    return scene.parent.name.removesuffix('.SAFE') if scene.name == 'manifest.safe' else scene.stem


def run_batch(raw: Path, output: Path, graph: Path, gpt: str,
              pixel_spacing: float, dem: str, overwrite: bool, dry_run: bool) -> None:
    if not raw.is_dir():
        raise FileNotFoundError(f'Input folder not found: {raw}')
    if not graph.is_file():
        raise FileNotFoundError(f'SNAP graph not found: {graph}')
    if not dry_run and shutil.which(gpt) is None and not Path(gpt).is_file():
        raise FileNotFoundError(f'GPT executable not found: {gpt}; set --gpt')

    scenes = find_scenes(raw)
    if not scenes:
        raise ValueError(f'No .SAFE or Sentinel-1 .zip files in {raw}')
    output.mkdir(parents=True, exist_ok=True)

    for number, scene in enumerate(scenes, 1):
        name = scene_id(scene)
        dest = output / f'{name}_sigma0_TC.tif'
        if dest.exists() and not overwrite:
            logging.info('SKIP existing %s', dest.name)
            continue
        cmd = [gpt, str(graph.resolve()),
               f'-Pinput={scene.resolve()}',
               f'-Poutput={dest.resolve()}',
               f'-Ppixel_spacing={pixel_spacing:g}',
               f'-Pdem={dem}',
               '-c', '2048M']
        logging.info('[%s/%s] %s', number, len(scenes), name)
        # logging.info('[{}/{}] {}', number, len(scenes), name)  # 必须确保你在项目初始化配置 logging.basicConfig() 时加上了 style='{' 参数，否则程序无法解析 {} 占位符。
        if dry_run:
            print(' '.join(f'"{x}"' if ' ' in x else x for x in cmd))
            continue
        log = output / f'{name}.log'
        try:
            # Sequential execution avoids multiple JVM jobs exhausting memory.
            with log.open('w', encoding='utf-8') as stream:
                subprocess.run(cmd, stdout=stream, stderr=subprocess.STDOUT,
                               check=True)
            if not dest.is_file() or dest.stat().st_size == 0:
                raise RuntimeError('SNAP exited normally but did not create a valid file')
            logging.info('OK -> %s', dest)
        except Exception:
            # An interrupted or unsuccessful run should not be considered processed.
            dest.unlink(missing_ok=True)
            logging.exception('FAILED processing %s; see %s', scene, log)
            raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--raw', type=Path, default=Path('raw'))
    parser.add_argument('--out', type=Path, default=Path('processed'))
    parser.add_argument('--graph', type=Path, default=Path('graphs/s1_grd_sigma0.xml'))  # -> SNAP XML graph defining preprocessing operations such as calibration and terrain correction.
    parser.add_argument('--gpt', default='gpt', help='gpt or full path to gpt.exe')      # -> Path or command for SNAP's Graph Processing Tool executable (gpt -> Graph Processing Tool)
    parser.add_argument('--pixel-spacing', type=float, default=10.0)                     # -> controls the output pixel spacing, not the actual spatial resolution of the sensor
    parser.add_argument('--dem', default='Copernicus 30m Global DEM')
    parser.add_argument('--overwrite', action='store_true')                              # -> if --overwrite, then it is Ture, otherwise it is False
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format='%(levelname)s %(message)s')
    run_batch(args.raw, args.out, args.graph, args.gpt,
              args.pixel_spacing, args.dem, args.overwrite, args.dry_run)


if __name__ == '__main__':
    main()
