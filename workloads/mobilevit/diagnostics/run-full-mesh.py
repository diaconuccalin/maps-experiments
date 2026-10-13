#!/usr/bin/env python3
"""Run the full MobileViT sequential full-mesh reference with layer progress."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import sys
import time


def write(path, data):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(data, indent=2) + '\n'); tmp.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mesh-root', required=True, type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--foreground', action='store_true')
    args = parser.parse_args()
    mesh = args.mesh_root.resolve()
    output = (args.output or mesh / 'maps-gvsoc-diagnostic' / ('full-mesh-' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ'))).resolve()
    output.mkdir(parents=True, exist_ok=True)
    if not args.foreground:
        with (output / 'monitor.log').open('w') as log:
            p = subprocess.Popen([sys.executable, '-u', str(Path(__file__).resolve()), '--mesh-root', str(mesh), '--output', str(output), '--foreground'], stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        (output / 'monitor.pid').write_text(str(p.pid)+'\n')
        print(json.dumps({'monitor_pid':p.pid,'output':str(output)},indent=2)); return 0
    sdk = Path('/home/zaino/repos/magia-sdk-v3')
    prepared = mesh.parent / 'prepared'
    app = output / 'application'; app.mkdir(exist_ok=True)
    reference = app / 'reference'
    shutil.copytree(Path(__file__).resolve().parents[1] / 'full-mesh', reference, dirs_exist_ok=True)
    source = (reference / 'main.c').read_text()
    source = source.replace('    kt_spatz_init((uint32_t)&_spatz_binary_start);', '    if (HID == 0) printf("FULL_MESH_INIT_BEGIN tiles=%u cycle=%u\\n", NUM_HARTS, read_cycle());\n    kt_spatz_init((uint32_t)&_spatz_binary_start);')
    source = source.replace('    if (HID == 0) window_start = read_cycle();', '    if (HID == 0) { printf("FULL_MESH_READY tiles=%u layers=%u tokens=%u cycle=%u\\n", NUM_HARTS, MVIT_NUM_LAYERS, MVIT_TOKENS, read_cycle()); window_start = read_cycle(); }')
    source = source.replace('            run_layer(&mobilevit_layers[i], token);', '            if (HID == 0) printf("FULL_MESH_LAYER_BEGIN token=%u layer=%u op=%u cycle=%u name=%s\\n", token, i, mobilevit_layers[i].op, read_cycle(), mobilevit_layers[i].name);\n            run_layer(&mobilevit_layers[i], token);')
    source = source.replace('            barrier(&fsync, &eu);', '            barrier(&fsync, &eu);\n            if (HID == 0) printf("FULL_MESH_LAYER_DONE token=%u layer=%u cycle=%u name=%s\\n", token, i, read_cycle(), mobilevit_layers[i].name);')
    (reference / 'main.c').write_text(source)
    (app / 'CMakeLists.txt').write_text(f'set(MVIT_DATA "{prepared}/full-mesh-data.bin")\nset(MVIT_PREPARED "{prepared}")\nset(MVIT_TOKENS 2)\nadd_subdirectory(reference)\n')
    env = os.environ.copy()
    env['PATH'] = str(Path(sys.executable).parent) + ':/home/zaino/repos/toolchains/pulp-toolchain/bin:' + env['PATH']
    env['LLVM_INSTALL_DIR'] = '/home/zaino/repos/toolchains/llvm20-root/usr/lib/llvm-20'
    env['LD_LIBRARY_PATH'] = '/home/zaino/repos/toolchains/llvm20-root/usr/lib/x86_64-linux-gnu:' + env.get('LD_LIBRARY_PATH','')
    env['CMAKE_BUILD_PARALLEL_LEVEL'] = '8'
    build = output / 'sdk-build'
    progress = {'state':'building','mesh':'32x32','tokens':2,'ready':False,'completed_layer_tokens':0,'completed_tokens':0,'errors':[]}
    write(output / 'progress.json', progress)
    cfg = sdk / 'targets/magia_v3/include/addr_map/tile_config.h'; saved = cfg.read_bytes()
    try:
        with (output / 'build.log').open('w') as log:
            for target in ['spatz_bootrom','onnx_mobilevit_full_mesh']:
                subprocess.run(['make','build','tiles=32',f'CMAKE_BUILDDIR={build}',f'maps_application_dir={app}',f'test={target}'], cwd=sdk, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
    except subprocess.CalledProcessError as e:
        progress.update(state='build_failed',exit_status=e.returncode);write(output/'progress.json',progress);return 1
    finally:
        cfg.write_bytes(saved)
    work = output / 'gvsoc'; work.mkdir(exist_ok=True)
    command = ['stdbuf','-oL','-eL',str(sdk/'gvsoc/install/bin/gvrun'),'--target','magia_v3','--param',f'binary={build}/bin/onnx_mobilevit_full_mesh','--work-dir',str(work),'--attr','magia_v3/n_tiles_x=32','--attr','magia_v3/n_tiles_y=32','--attr',f'magia_v3/spatz_romfile={build}/bin/bootrom/spatz_init.bin','run']
    p = subprocess.Popen(command,cwd=sdk,env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1,start_new_session=True)
    (output/'simulator.pid').write_text(str(p.pid)+'\n')
    progress.update(state='running',simulator_pid=p.pid,last_event_at=None,active_layer=None)
    write(output/'progress.json',progress)
    print(f'RUN_STARTED simulator_pid={p.pid} tiles=1024 tokens=2',flush=True)
    with (output/'full-mesh.log').open('w',buffering=1) as log, (output/'layer-events.jsonl').open('w',buffering=1) as events:
        for line in p.stdout:
            log.write(line)
            m = re.search(r'(FULL_MESH_\w+|REFERENCE_COMPLETION|REFERENCE_VALIDATION)\b(.*)',line)
            if not m: continue
            marker,rest = m.groups(); data=dict(re.findall(r'(\w+)=([^\s]+)',rest))
            stamp = datetime.now(timezone.utc).isoformat()
            record={'observed_at':stamp,'marker':marker,'details':data};events.write(json.dumps(record)+'\n')
            progress.update(last_event_at=stamp,last_observed_cycle=int(data.get('cycle',progress.get('last_observed_cycle',0))))
            if marker=='FULL_MESH_READY':progress.update(ready=True,layers=int(data['layers']))
            elif marker=='FULL_MESH_LAYER_BEGIN':progress['active_layer']=data
            elif marker=='FULL_MESH_LAYER_DONE':progress['completed_layer_tokens']+=1;progress['active_layer']=None
            elif marker=='REFERENCE_COMPLETION':progress['completed_tokens']+=1
            write(output/'progress.json',progress)
            print(line.strip(),flush=True)
    code=p.wait();progress.update(state='completed' if code==0 and progress['completed_tokens']==2 else 'simulator_failed',exit_status=code)
    write(output/'progress.json',progress);print(f'RUN_FINISHED state={progress["state"]} exit_status={code}',flush=True)
    return 0 if progress['state']=='completed' else 1

if __name__ == '__main__':raise SystemExit(main())
