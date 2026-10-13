#!/usr/bin/env python3
"""Test the original first two MobileViT stages and their FIFO links on 4x4.

Uses the SDK's actual asynchronous FIFO runner. Preserves all stage 0 -> stage 1
transfers, producer indices, strided layouts, input data, and two token slots.
Only sends to later stages are removed. Prints operation errors before traps.
"""

import importlib.util
import json
from pathlib import Path
import re

import numpy as np

spec = importlib.util.spec_from_file_location('first_stage', Path(__file__).with_name('isolate-first-stage.py'))
first = importlib.util.module_from_spec(spec)
spec.loader.exec_module(first)
generate_first = first.generate


def descriptor_blocks(text):
    """Split generated array entries, without changing nested descriptors."""
    blocks, depth, begin = [], 0, None
    for index, character in enumerate(text):
        if character == '{':
            if depth == 0:
                begin = index
            depth += 1
        elif character == '}':
            depth -= 1
            if depth == 0:
                blocks.append(text[begin:index + 1])
    assert depth == 0
    return blocks


def generate(mesh_root, output):
    app = generate_first(mesh_root, output)
    original = mesh_root / 'maps-application'
    ids = list(range(416, 424)) + [228, 229, 230, 261, 262]
    mapping = dict(zip(ids, range(13)))
    consumers = set(ids[8:])
    manifest = json.loads((output / 'manifest.json').read_text())
    full_output = np.zeros((2, 16, 128, 128), dtype=np.float16)
    full_input = np.fromfile(original / 'data/maps_mobilevit.input_input.bin', dtype='<f2').reshape(2, 3, 256, 256)
    input_refs, input_counts, input_asm, input_declarations = [], [], [], []
    for item in manifest:
        tile = item['test_tile']
        origin = item['origins'][4][2]
        full_output[:, :, origin:origin + 16, :] = np.fromfile(
            app / f'golden_{tile}_output.bin', dtype='<f2').reshape(2, 16, 16, 128)
        input_origin = item['origins'][2]
        input_shape = item['shapes'][2]
        shard = full_input[:, :, input_origin[2]:input_origin[2] + input_shape[2], :].copy()
        path = app / f'golden_{tile}_l2_input.bin'
        shard.astype('<f2').tofile(path)
        symbol = f'golden_{tile}_l2_input'
        input_declarations.append(f'extern const uint16_t {symbol}[];')
        input_refs.append(symbol)
        input_counts.append(str(shard.size // 2) + 'u')
        input_asm.append(f'\n.section .l2_bulk.{symbol},"a",@progbits\n.balign 16\n.global {symbol}\n{symbol}:\n.incbin "{path}"\n')
    main = (app / 'main.c').read_text()
    main = main.replace('maps_mobilevit_first_stage_task_bin.h', 'maps_mobilevit_first_two_stages_task_bin.h')
    main = main.replace('plans[8]', 'plans[13]').replace('golden[8][2]', 'golden[13][2]')
    plan_names, golden_names, declarations, counts = [], [], [], []
    asm = (app / 'data.S').read_text() + ''.join(input_asm)
    for tile, old in enumerate(ids):
        source = (original / f'src/tiles/tile_{old:04d}.c').read_text()
        sends_name = f'maps_mobilevit_tile_{old}_sends'
        prefix, rest = source.split(f'static const fifo_send_desc_t {sends_name}[] = {{', 1)
        sends_text, suffix = rest.split('\n};', 1)
        sends = [block for block in descriptor_blocks(sends_text)
                 if int(re.search(r'\.dst_hartid = (\d+)', block).group(1)) in consumers]
        if sends:
            text = prefix + f'static const fifo_send_desc_t {sends_name}[] = {{\n' + ',\n'.join(sends) + ',\n};' + suffix
        else:
            text = prefix + suffix
        text = re.sub(r'\.num_sends = \d+u,', f'.num_sends = {len(sends)}u,', text)
        if not sends:
            text = re.sub(r'\.sends = [^,]+,', '.sends = 0,', text)
        text = re.sub(r'(\.(?:owner_hartid|src_hartid|dst_hartid|hartid) = )(\d+)(u?)(,)',
                      lambda m: m[1] + str(mapping[int(m[2])]) + m[3] + m[4], text)
        (app / f'tile_{tile}.c').write_text(text)
        plan_names.append(f'&maps_mobilevit_tile_{old:04d}_plan')
        if tile < 8:
            golden_names.append(f'{{golden_{tile}_columns, golden_{tile}_output}}')
            counts.append('{55296u, 32768u}')
            manifest[tile]['retained_fifo_sends'] = len(sends)
            continue
        slices = prefix.split(f'maps_mobilevit_tile_{old}_slices[] = {{', 1)[1].split('\n};', 1)[0]
        shape = list(map(int, re.search(r'\.shape = \{([^}]+)\}', slices).group(1).split(',')))
        origin = list(map(int, re.search(r'\.full_offset = \{([^}]+)\}', slices).group(1).split(',')))
        assert shape[:2] == [1, 16] and shape[3] == 128
        count = 16 * shape[2] * 128
        x = full_output[:, :, origin[2]:origin[2] + shape[2], :].copy()
        mathematical = (1.0 / (1.0 + np.exp(-x.astype(np.float64)))).astype(np.float16)
        # The SDK intentionally uses the same fast sigmoid approximation as
        # its Spatz kernel. Replay each binary16 rounding and reinterpretation;
        # a true mathematical sigmoid is a separate accuracy diagnostic.
        clamped = np.maximum(x.astype(np.float32), np.float32(-11.0))
        with np.errstate(over='ignore'):
            scaled = (-clamped * np.float32(1477.0)).astype(np.float16)
        biased = (scaled.astype(np.float32) + np.float32(15360.0)).astype(np.float16)
        exponential = np.clip(biased.astype(np.float32), 0, 65535).astype(np.uint16).view(np.float16)
        denominator = (np.float32(1.0) + exponential.astype(np.float32)).astype(np.float16)
        y = (np.float32(1.0) / denominator.astype(np.float32)).astype(np.float16)
        assert np.all(np.isfinite(y)) and np.all((y >= 0) & (y <= 1))
        for name, data in [('input', x), ('output', y)]:
            symbol = f'golden_{tile}_{name}'
            path = app / f'{symbol}.bin'
            data.astype('<f2').tofile(path)
            asm += f'\n.section .l2_bulk.{symbol},"a",@progbits\n.balign 16\n.global {symbol}\n{symbol}:\n.incbin "{path}"\n'
            declarations.append(f'extern const uint16_t {symbol}[];')
        declarations.append(f'extern const fifo_tile_plan_t maps_mobilevit_tile_{old:04d}_plan;')
        golden_names.append(f'{{golden_{tile}_input, golden_{tile}_output}}')
        counts.append(f'{{{count}u, {count}u}}')
        manifest.append({'original_tile': old, 'test_tile': tile, 'stage': 1,
                         'shape': shape, 'origin': origin, 'elements_per_token': count,
                         'sigmoid_reference_max_absolute_error_vs_mathematical':
                         float(np.max(np.abs(y.astype(np.float64) - mathematical.astype(np.float64)))),
                         'retained_fifo_sends': len(sends)})
    (app / 'data.S').write_text(asm)
    (output / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    main = re.sub(r'static const fifo_tile_plan_t \*const plans\[13\] = \{[^\n]+',
                  'static const fifo_tile_plan_t *const plans[13] = {' + ', '.join(plan_names) + '};', main)
    main = re.sub(r'static const uint16_t \*const golden\[13\]\[2\] = \{[^\n]+',
                  'static const uint16_t *const golden[13][2] = {' + ', '.join(golden_names) + '};', main)
    main = main.replace('static const fifo_tile_plan_t *const plans', '\n'.join(declarations) + '\nstatic const fifo_tile_plan_t *const plans')
    main = main.replace('static volatile uint32_t errors',
                        'static const uint32_t counts[13][2] = {' + ', '.join(counts) + '};\nstatic volatile uint32_t errors')
    # Instrument the real dispatcher, then include the unchanged FIFO runner.
    main = main.replace('#include "utils/maps_operations.h"', '''
#define maps_execute_operation isolated_base_execute_operation
#include "utils/maps_operations.h"
#undef maps_execute_operation
static uint32_t isolated_check_input(const tile_plan_t *plan, const op_desc_t *op, uint32_t slot);
static inline int maps_execute_operation(const tile_plan_t *plan, const op_desc_t *op,
                                          uint32_t slot, maps_operation_runtime_t *runtime)
{
    if (op->kind == OP_IM2COL) isolated_check_input(plan, op, slot);
    uint32_t start = maps_read_cycle();
    printf("TWO_STAGE_OP_BEGIN tile=%u slot=%u kind=%u cycle=%u\\n", plan->hartid, slot, op->kind, start);
    int result = isolated_base_execute_operation(plan, op, slot, runtime);
    printf("TWO_STAGE_OP_DONE tile=%u slot=%u kind=%u result=%d cycles=%u\\n",
           plan->hartid, slot, op->kind, result, maps_read_cycle() - start);
    if (result) printf("ISOLATED_OPERATION_FAILURE tile=%u slot=%u kind=%u result=%d\\n",
                       plan->hartid, slot, op->kind, result);
    return result;
}
''')
    main = main.replace('static const fifo_tile_plan_t *const plans',
                        '\n'.join(input_declarations) + '\n'
                        + 'static const uint16_t *const input_golden[8] = {' + ', '.join(input_refs) + '};\n'
                        + 'static const uint32_t input_counts[8] = {' + ', '.join(input_counts) + '};\n'
                        + 'static const fifo_tile_plan_t *const plans')
    main = main.replace('int main(void)', '''static uint32_t isolated_check_input(const tile_plan_t *plan, const op_desc_t *op, uint32_t slot)
{
    uint32_t tile = plan->hartid, count = input_counts[tile];
    const uint16_t *data = (const uint16_t *)local_subslice_addr(plan, &op->inputs[0], slot);
    uint32_t bad = check(data, input_golden[tile] + slot * count, count, tile, slot, 2u);
    errors[tile] += bad;
    if (bad) printf("ISOLATED_INPUT_FAILURE tile=%u slot=%u errors=%u\\n", tile, slot, bad);
    return bad;
}

int main(void)''')
    main = main.replace('hartid < 8', 'hartid < 13')
    begin = main.index('        /* Same read/compute ordering')
    end = main.index('\n    }\n    fsync_sync_global', begin)
    main = main[:begin] + '''        printf("TWO_STAGE_RUN_BEGIN tile=%u cycle=%u\\n", hartid, maps_read_cycle());
        maps_fifo_run_tile_tokens(&plan, 2u, &idma, &eu);
        printf("TWO_STAGE_RUN_DONE tile=%u cycle=%u\\n", hartid, maps_read_cycle());
        for (uint32_t token = 0; token < 2; ++token) {
            uint32_t slot = maps_token_slot((const tile_plan_t *)&plan, token);
            for (uint32_t which = 0; which < 2; ++which) {
                const subslice_desc_t *sub = hartid < 8
                    ? &plan.ops[which].outputs[0]
                    : (which == 0 ? &plan.ops[0].inputs[0] : &plan.ops[0].outputs[0]);
                const uint16_t *data = (const uint16_t *)local_subslice_addr((const tile_plan_t *)&plan, sub, slot);
                uint32_t count = counts[hartid][which];
                errors[hartid] += check(data, golden[hartid][which] + token * count,
                                       count, hartid, token, which);
            }
        }''' + main[end:]
    main = main.replace('FIRST_STAGE_', 'TWO_STAGE_').replace('active_tiles=8 tokens=2', 'active_tiles=13 tokens=2')
    (app / 'main.c').write_text(main)
    cmake = (app / 'CMakeLists.txt').read_text().replace('maps_mobilevit_first_stage', 'maps_mobilevit_first_two_stages')
    cmake = cmake.replace('  tile_7.c\n)', '  tile_7.c\n' + ''.join(f'  tile_{i}.c\n' for i in range(8, 13)) + ')')
    (app / 'CMakeLists.txt').write_text(cmake)
    return app


def summarize(output):
    log = (output / 'maps.log').read_text()
    mode = 'blocking' if 'TWO_STAGE_RUN_MODE blocking' in log else 'asynchronous'
    failures = [dict(zip(('tile', 'slot', 'kind', 'result'), map(int, m))) for m in re.findall(
        r'ISOLATED_OPERATION_FAILURE tile=(\d+) slot=(\d+) kind=(\d+) result=(-?\d+)', log)]
    checks = [dict(zip(('tile', 'token', 'op', 'elements', 'errors', 'cycle'), map(int, m))) for m in re.findall(
        r'TWO_STAGE_CHECK tile=(\d+) token=(\d+) op=(\d+) elements=(\d+) errors=(\d+) cycle=(\d+)', log)]
    completed = {(int(t), int(n)) for t, n in re.findall(r'MAPS_TOKEN_DONE tile=(\d+) token=(\d+)', log)}
    input_checked = any(c['op'] == 2 for c in checks)
    expected = {(t, n, op) for t in range(13) for n in range(2)
                for op in range(3 if input_checked and t < 8 else 2)}
    passed = (not failures and len(checks) == len(expected) and
              {(c['tile'], c['token'], c['op']) for c in checks} == expected and
              completed == {(t, n) for t in range(13) for n in range(2)} and
              all(c['errors'] == 0 for c in checks) and
              'TWO_STAGE_RESULT errors=0 active_tiles=13 tokens=2' in log)
    summary = {'passed': passed, 'mesh': '4x4', 'active_tiles': 13, 'tokens': 2,
               'fifo_mode': mode,
               'l2_input_checked': input_checked,
               'completed_tile_tokens': len(completed), 'completed': sorted(completed),
               'checked_elements': sum(c['elements'] for c in checks),
               'mismatches': sum(c['errors'] for c in checks),
               'operation_failures': failures, 'checks': checks,
               'scope': f'Original first two stages, actual {mode} FIFO runner and stage 0 -> stage 1 transfers. Sends to stage 2 removed. Comparisons exact against FP16 arithmetic replay, including SDK fast sigmoid approximation.'}
    (output / 'statistics.json').write_text(json.dumps(summary, indent=2) + '\n')
    return summary


if __name__ == '__main__':
    first.__doc__ = __doc__
    first.generate = generate
    first.summarize = summarize
    first.main('maps_mobilevit_first_two_stages', 'first-two-stages-4x4')
