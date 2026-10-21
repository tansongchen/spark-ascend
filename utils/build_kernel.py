"""Compile checked-in Ascend C sources and preserve their loader manifest."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from utils.kernel_catalog import KERNELS, manifest_for


def source_dependencies(source):
    """Collect local quoted includes; toolkit includes are resolved by bisheng."""
    files = {source.resolve()}
    for include in re.findall(r'^\s*#include\s*"([^"]+)"', source.read_text(), re.M):
        header = source.parent / include
        if header.is_file():
            files.update(source_dependencies(header))
    return files


def build(name, output):
    kernel = KERNELS[name]
    output = Path(output).resolve()
    if not output.is_relative_to('/data1/t00995472'):
        raise ValueError('构建产物必须放在 /data1/t00995472/ 下')
    if output.exists() and any(output.iterdir()):
        raise ValueError(f'请使用空构建目录，避免覆盖已有库：{output}')
    if not os.environ.get('ASCEND_HOME_PATH') or not shutil.which('bisheng'):
        raise RuntimeError('请先 source /data1/t00995472/spark/activate_ascend.sh')
    output.mkdir(parents=True, exist_ok=True)
    for source in source_dependencies(ROOT / 'kernels' / kernel.source):
        shutil.copy2(source, output / source.name)
    env = os.environ.copy()
    env.pop('DEBUG', None)
    def run(cmd):
        print('+', ' '.join(map(str, cmd)), flush=True)
        subprocess.run(list(map(str, cmd)), env=env, check=True)
    run(['bisheng', '--npu-arch=dav-3510', '-fPIC', '-c', output/kernel.source, '-o', output/'device_pic.o'])
    run(['bisheng', '--npu-arch=dav-3510', '-shared', output/'device_pic.o',
         '-o', output/kernel.library, '-L'+env['ASCEND_HOME_PATH']+'/lib64', '-lascendcl'])
    manifest = manifest_for(name)
    manifest['files_sha256'] = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                for p in sorted(output.iterdir()) if p.suffix in ('.asc', '.h', '.so')}
    manifest['source_sha256'] = {k: v for k, v in manifest['files_sha256'].items() if not k.endswith('.so')}
    if 'cpu_codec' in manifest:
        manifest['cpu_codec_sha256'] = hashlib.sha256((ROOT/manifest['cpu_codec']).read_bytes()).hexdigest()
    (output/'execution.json').write_text(json.dumps(manifest, indent=2)+'\n')
    print(f'Library: {output/kernel.library}\nManifest: {output/"execution.json"}')


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('--kernel', choices=KERNELS, default='stage_prefetch')
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    build(a.kernel, a.output)


if __name__ == '__main__':
    main()
