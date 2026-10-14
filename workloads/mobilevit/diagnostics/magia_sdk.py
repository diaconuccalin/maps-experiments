"""Build and launch helpers shared by the MobileViT diagnostics.

Mirror run-experiment.sh: the SDK comes from MAGIA_SDK_ROOT (default
<checkout>/magia-sdk-v3) and the toolchains from the environment (source
maps-env.sh first). The SDK is built without the PULP cluster (PULP_CORES,
default 0), while GVSoC's MAGIA v3 target needs at least one PULP core per
tile (GVSOC_PULP_CORES, default 8); they stay idle.
"""

import os
from pathlib import Path
import subprocess
import sys

CHECKOUT_ROOT = Path(__file__).resolve().parents[4]
PULP_CORES = os.environ.get('PULP_CORES', '0')
GVSOC_PULP_CORES = os.environ.get('GVSOC_PULP_CORES', '8')


def default_sdk_root():
    return Path(os.environ.get('MAGIA_SDK_ROOT', CHECKOUT_ROOT / 'magia-sdk-v3'))


def build_env(sdk):
    env = os.environ.copy()
    if 'LLVM_INSTALL_DIR' not in env:
        for candidate in (sdk / 'llvm/install',
                          CHECKOUT_ROOT / 'toolchains/llvm20-root/usr/lib/llvm-20',
                          Path('/usr/lib/llvm-20'), Path('/usr/lib/llvm-18')):
            if all((candidate / 'bin' / tool).exists()
                   for tool in ('clang', 'llvm-objcopy', 'llvm-objdump')):
                env['LLVM_INSTALL_DIR'] = str(candidate)
                break
        else:
            raise SystemExit('set LLVM_INSTALL_DIR to a Spatz-capable LLVM installation')
    runtime_libs = Path(env['LLVM_INSTALL_DIR']).parent / 'x86_64-linux-gnu'
    if runtime_libs.is_dir():
        env['LD_LIBRARY_PATH'] = str(runtime_libs) + ':' + env.get('LD_LIBRARY_PATH', '')
    env['PATH'] = str(Path(sys.executable).parent) + ':' + env['PATH']
    env.setdefault('CMAKE_BUILD_PARALLEL_LEVEL', '8')
    return env


def build(sdk, env, log, tiles, build_dir, application, targets, options=()):
    """Build GVSoC for this mesh, then each SDK target of the application."""
    subprocess.run(['make', 'gvsoc', f'tiles={tiles}', f'pulp_cores={GVSOC_PULP_CORES}'],
                   cwd=sdk, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
    for target in targets:
        subprocess.run(['make', 'build', f'tiles={tiles}', f'pulp_cores={PULP_CORES}',
                        f'CMAKE_BUILDDIR={build_dir}', f'maps_application_dir={application}',
                        *options, f'test={target}'],
                       cwd=sdk, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)


def gvrun_command(sdk, build_dir, binary, work, tiles, options=()):
    return ['stdbuf', '-oL', '-eL', str(sdk / 'gvsoc/install/bin/gvrun'),
            '--target', f'magia_v3:n_tiles_x={tiles},n_tiles_y={tiles},nb_pulp_cores={GVSOC_PULP_CORES}',
            '--param', f'binary={build_dir}/bin/{binary}', '--work-dir', str(work),
            '--attr', f'magia_v3/spatz_romfile={build_dir}/bin/bootrom/spatz_init.bin',
            *options, 'run']
