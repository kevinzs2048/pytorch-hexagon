"""Validate a compile-only wrapper ownership relink without approving inference."""
import json
from pathlib import Path
from hexagon_generate_reuse import sha


def verify_owned_wrapper_relink(compiled, original=None):
    proof=compiled.get('wrapper_relink',{})
    if (compiled.get('returned_buffer_policy')!='owned-DDR-single-call'
            or proof.get('model_graph_recompiled') is not False or proof.get('weights_changed') is not False
            or proof.get('process_exit_code')!=0):
        raise ValueError('Missing completed, arithmetic-preserving ownership relink')
    source_path=Path(proof['source_result'])
    if sha(source_path)!=proof['source_result_sha256']:
        raise ValueError('Ownership relink source drift')
    source=json.loads(source_path.read_text())
    if original is not None:
        for key in ('capacity','precision','checkpoint','historical_parameter_sha256'):
            if original[key]!=source[key]: raise ValueError('Saved reference model differs')
        if ([x['sha256'] for x in original['graphs'][0]['libraries']]
                != [x['sha256'] for x in source['graphs'][0]['libraries']]):
            raise ValueError('Reference uses a different original compiled model')
    for key in ('capacity','precision','checkpoint','historical_parameter_sha256','options',
                'fp32_sum_policy','fp32_attention_dot_policy','fp32_softmax_policy','scalar_exp_policy','rotary_policy'):
        if compiled[key]!=source[key]: raise ValueError('Relink changed model or numerical context: '+key)
    for key in ('fp32_context_dot_policy','attention_control','native_softmax_control'):
        if compiled.get(key)!=source.get(key):
            raise ValueError('Relink changed capacity-64 provenance: '+key)
    graph=compiled['graphs'][0]
    if (len(compiled['graphs'])!=1 or graph['graph_sha256']!=source['graphs'][0]['graph_sha256']
            or graph.get('returned_buffer_policy')!='owned-DDR-single-call'
            or graph['libraries'][0]['sha256']!=proof['main_library_sha256']
            or [x['sha256'] for x in graph['libraries'][1:]]!=[x['sha256'] for x in source['graphs'][0]['libraries'][1:]]):
        raise ValueError('Relink changed graph or weights')
    if proof['old_model_libraries']!=source['graphs'][0]['libraries']:
        raise ValueError('Original library set drift')
    if len(proof['objects'])!=2:
        raise ValueError('Expected original model and selective Softmax objects only')
    command=proof['link_command']; original_command=source['scalar_exp_binding']['link_command']
    old_dir=Path(source['graphs'][0]['libraries'][0]['path']).parent
    # Execution-only reuse may hardlink DSOs into another evidence directory;
    # the actual compile command remains bound to its original wrapper folder.
    new_dir=Path(proof['wrapper']['path']).parent
    expected=[str(new_dir/Path(p).name) if p.startswith(str(old_dir)+'/') else
        '-L'+str(new_dir) if p=='-L'+str(old_dir) else p for p in original_command]
    expected.insert(1,'-I'+str(Path(proof['owner_header']['path']).parent))
    if command!=expected:
        raise ValueError('Unexpected link policy or numerical object substitution')
    for item in proof['objects']:
        if item['source_path'] not in original_command or item['path'] not in command:
            raise ValueError('Unbound model object')
        if sha(item['path'])!=item['sha256'] or sha(item['source_path'])!=item['sha256']:
            raise ValueError('Model object bytes changed')
    for item in (proof['wrapper'],proof['owner_header'],*graph['libraries']):
        if sha(item['path'])!=item['sha256']:
            raise ValueError('Relinked artifact drift')
    if (graph['original_wrapper_path']!=proof['wrapper']['path']
            or graph['original_wrapper_sha256']!=proof['wrapper']['sha256']):
        raise ValueError('Unbound ownership wrapper')
    from hexagon_buffer_binding import validate_buffer_binding
    validate_buffer_binding(compiled)
    return proof
