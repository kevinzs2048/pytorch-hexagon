"""Exact capacity1024 compiler-cache admission, not generation acceptance.

Bind each fresh native graph to its actual Linalg capture and compiled ABI.
DSP model/transport proofs remain separate mandatory runtime prerequisites.
No cache-miss compilation or CPU fallback is implemented here.
"""

from hexagon_extended_kv_layout import ExtendedKVLayout


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
