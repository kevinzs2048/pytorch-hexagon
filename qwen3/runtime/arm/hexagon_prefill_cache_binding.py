"""Bind each declared full prefill/decode graph to its actual compiled ABI."""
from hexagon_prefill1024_layout_candidate import Prefill1024LayoutCandidate


def check_graph_binding(compiled,capture,*,context,query_tokens,layout):
    if type(query_tokens) is not int or query_tokens not in (1,2,3,4):
        raise ValueError('Only declared1/2/3/4-token full graphs')
    old=query_tokens in (1,2)
    status='native1024_model_machine_code_compiled_NOT_executed' if old else 'prefill_model_machine_code_compiled_NOT_executed'
    capture_status='extended_native_Linalg_exported_NOT_machine_code_or_executed' if old else 'prefill_native_Linalg_exported_NOT_machine_code_or_executed'
    if (compiled.get('status')!=status or capture.get('status')!=capture_status
            or compiled.get('capacity')!=1024 or capture.get('capacity')!=1024
            or compiled.get('query_tokens')!=query_tokens or capture.get('query_tokens')!=query_tokens
            or compiled.get('bindings',{}).get('context')!=context or capture.get('context')!=context
            or len(capture.get('graphs',[]))!=1):
        raise ValueError('Same complete native capture/compiler/context required')
    graph=capture['graphs'][0]
    ins,outs=Prefill1024LayoutCandidate(query_tokens=query_tokens).signature()
    updates={str(i):j for j,i in enumerate([3,4,*range(6,60)])}
    if (graph.get('inputs')!=ins or graph.get('outputs')!=outs or compiled.get('inputs')!=ins
            or compiled.get('outputs')!=outs or compiled.get('layout')!=layout
            or len(graph.get('parameters',[]))!=509 or compiled.get('parameters')!=graph['parameters']
            or compiled.get('storage')!=graph.get('storage')
            or graph.get('input_update_outputs')!=updates or compiled.get('input_update_outputs')!=updates):
        raise ValueError('Complete parameter/storage/cache writeback binding differs')
    return graph
