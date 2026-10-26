"""Exact capacity1024 compiler-cache admission, not generation acceptance.

Bind each fresh native graph to its actual Linalg capture and compiled ABI.
DSP model/transport proofs remain separate mandatory runtime prerequisites.
No cache-miss compilation or CPU fallback is implemented here.
"""
import json
from pathlib import Path

from hexagon_generate_reuse import sha
from hexagon_extended_kv_layout import ExtendedKVLayout
from hexagon_buffer_binding import local_buffer_io


def read_bound(row):
    path = Path(row['path'])
    before = sha(path)
    if before != row['sha256']:
        raise ValueError('Compiler-cache dependency changed')
    result = json.loads(path.read_text())
    if sha(path) != before:
        raise ValueError('Compiler-cache dependency still changing')
    return result


def check_graph_binding(compiled, capture, *, context, query_tokens, layout):
    if (compiled.get('status') != 'native1024_model_machine_code_compiled_NOT_executed'
            or capture.get('status') != 'extended_native_Linalg_exported_NOT_machine_code_or_executed'
            or compiled.get('capacity') != 1024 or capture.get('capacity') != 1024
            or compiled.get('query_tokens') != query_tokens or capture.get('query_tokens') != query_tokens
            or compiled.get('bindings', {}).get('context') != context
            or capture.get('context') != context or len(capture.get('graphs', [])) != 1):
        raise ValueError('Same complete capacity1024 native capture/compiler/context required')
    graph = capture['graphs'][0]
    ins, outs = ExtendedKVLayout(1024, query_tokens=query_tokens).signature()
    expected_updates = {str(i): j for j, i in enumerate([3, 4, *range(6, 60)])}
    if (graph.get('inputs') != ins or graph.get('outputs') != outs
            or compiled.get('inputs') != ins or compiled.get('outputs') != outs
            or compiled.get('layout') != layout
            or len(graph.get('parameters', [])) != 509
            or compiled.get('parameters') != graph.get('parameters')
            or compiled.get('storage') != graph.get('storage')
            or graph.get('input_update_outputs') != expected_updates
            or compiled.get('input_update_outputs') != expected_updates):
        raise ValueError('Native parameters, lossless storage or full cache-writeback ABI differs')
    return graph


def load_native1024_cache_binding(linked_path, context_path):
    linked_path, context_path = Path(linked_path), Path(context_path)
    linked_row = dict(path=str(linked_path), sha256=sha(linked_path))
    context = dict(path=str(context_path), sha256=sha(context_path))
    linked = read_bound(linked_row)
    if (linked.get('status') != 'native1024_shared_entries_linked_NOT_executed'
            or linked.get('capacity') != 1024
            or linked.get('model_arithmetic_objects_unchanged') is not True
            or len(linked.get('compiled', [])) != 2):
        raise ValueError('Isolated two-shape capacity1024 shared linkage required')
    built, captures = {}, {}
    bio = local_buffer_io()
    for row in linked['compiled']:
        compiled = read_bound(row)
        n = compiled.get('query_tokens')
        if type(n) is not int or n not in (1, 2) or n in built:
            raise ValueError('Exactly one artifact per native shape required')
        capture = read_bound(compiled['bindings']['capture'])
        layout = bio.make_layout(*ExtendedKVLayout(1024, query_tokens=n).signature())
        check_graph_binding(compiled, capture, context=context, query_tokens=n, layout=layout)
        if linked['layouts'].get(str(n)) != layout:
            raise ValueError('Shared entry layout differs from actual compiled graph')
        for source in compiled['sources'] + capture['sources']:
            if sha(source['path']) != source['sha256']:
                raise ValueError('Native compiler/framework source changed')
        serializer = [r for r in compiled['sources'] if Path(r['path']).name == 'hexagon_buffer_io.py']
        if len(serializer) != 1 or sha(bio.__file__) != serializer[0]['sha256']:
            raise ValueError('Actual native serializer differs from compiled ABI')
        read_bound(capture['rotary_control'])
        built[n], captures[n] = compiled, capture
    if set(built) != {1, 2} or captures[1]['rotary_control'] != captures[2]['rotary_control']:
        raise ValueError('Both shapes must use the same original position tables')
    if (built[1]['parameters'] != built[2]['parameters']
            or built[1]['storage'] != built[2]['storage']):
        raise ValueError('Native shapes must share identical original weights and storage policy')
    if sha(linked_path) != linked_row['sha256'] or sha(context_path) != context['sha256']:
        raise ValueError('Compiler-cache root changed during admission')
    return dict(linked=linked, linked_binding=linked_row, context=context,
                built=built, captures=captures, generation_accepted=False)

