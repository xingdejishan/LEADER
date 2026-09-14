from pathlib import Path
import requests
import zipfile

root = Path(__file__).resolve().parents[4] / 'research/projection_audit_assets'
root.mkdir(parents=True, exist_ok=True)
base = 'https://s3.us-east-2.amazonaws.com/nclt.perl.engin.umich.edu/'
for name in ['ladybug3_calib/cam_params.zip', 'python/project_vel_to_cam.py', 'python/undistort.py',
             'ladybug3_calib/U2D_Cam5_1616X1232.txt']:
    target = root / Path(name).name
    if target.exists():
        continue
    with requests.get(base + name, stream=True, timeout=(20, 90)) as response:
        response.raise_for_status()
        with target.with_suffix('.pending').open('wb') as output:
            for block in response.iter_content(1024 * 1024):
                output.write(block)
    target.with_suffix('.pending').replace(target)
    print(target.name, target.stat().st_size, flush=True)
with zipfile.ZipFile(root / 'cam_params.zip') as archive:
    for name in archive.namelist():
        if name.endswith('.csv'):
            (root / Path(name).name).write_bytes(archive.read(name))
