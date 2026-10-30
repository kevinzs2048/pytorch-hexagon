"""Byte/provenance guards for compiled floats, independent of tensor binding."""
import copy
import hashlib
import math
from pathlib import Path
import pytest
import torch
from models.qwen3.runtime.support.guarded_module_constants import (GuardedModuleConstants,SCHEMA,
    constant_signature,validate_constant_rows,validate_capture_constants)
from models.qwen3.runtime.support import guarded_module_constants as constants


def norm():
    value=torch.nn.Module();value.variance_epsilon=1e-6
    return value


def model():
    root=torch.nn.Module();root.model=torch.nn.Module()
    root.model.norm=norm();root.model.rotary_emb=torch.nn.Module()
    root.model.rotary_emb.attention_scaling=1.0
    root.model.layers=torch.nn.ModuleList()
    for i in range(28):
        layer=torch.nn.Module();layer.input_layernorm=norm();layer.post_attention_layernorm=norm()
        layer.self_attn=torch.nn.Module();layer.self_attn.scaling=0.125
        layer.self_attn.q_norm=norm();layer.self_attn.k_norm=norm()
        root.model.layers.append(layer)
    return root


def test_all_native_FP64_values_match_tensor_representation_and_owned_ledger():
    m=model();guard=GuardedModuleConstants(m);guard.check()
    assert len(guard.records)==142
    for row in guard.records:
        value=m.get_submodule(row['module']).__dict__[row['attribute']]
        tensor=torch.tensor(value,dtype=torch.float64).reshape(1).view(torch.uint8).numpy()
        assert bytes(tensor).hex()==row['hex_le']
        assert hashlib.sha256(bytes(tensor)).hexdigest()==row['sha256']
    ledger=guard.records;ledger[0]['hex_le']='00'*8
    assert guard.records[0]['hex_le']!=ledger[0]['hex_le']
    guard.check()


@pytest.mark.parametrize('mutation',['next_float','tensor','nan','inf','missing','owner'])
def test_runtime_changes_rejected_before_execution(mutation):
    m=model();guard=GuardedModuleConstants(m);owner=m.model.layers[13].self_attn
    if mutation=='next_float':owner.scaling=math.nextafter(owner.scaling,math.inf)
    elif mutation=='tensor':owner.scaling=torch.tensor(owner.scaling,dtype=torch.float64)
    elif mutation=='nan':owner.scaling=math.nan
    elif mutation=='inf':owner.scaling=math.inf
    elif mutation=='missing':del owner.scaling
    else:
        replacement=torch.nn.Module();replacement.scaling=owner.scaling
        replacement.q_norm=owner.q_norm;replacement.k_norm=owner.k_norm
        m.model.layers[13].self_attn=replacement
    with pytest.raises(ValueError,match='before DSP'):guard.check()


def test_signed_zero_and_compiler_host_mismatch_rejected():
    m=model();m.model.layers[27].self_attn.scaling=0.0
    guard=GuardedModuleConstants(m);rows=guard.records
    m.model.layers[27].self_attn.scaling=-0.0
    with pytest.raises(ValueError,match='before DSP'):guard.check()
    with pytest.raises(ValueError,match='compiler-host|Compiler-host'):GuardedModuleConstants(m,rows)


def test_complete_selected_model_content_matches_legacy_and_detects_changes():
    from models.qwen3.runtime.support.guarded_module_constants import selected_model_content_signature
    from hexagon_module_scalar_binding import binding_content_signature
    rows=GuardedModuleConstants(model()).records
    tensors=[dict(source='parameter'+str(i),attributes=[],
        spec=dict(shape=[1],dtype='torch.int8',stride=[1]),bytes=1,
        sha256=hashlib.sha256(bytes([i%256])).hexdigest()) for i in range(509)]
    legacy=[]
    for row in rows:
        value=model().get_submodule(row['module']).__dict__[row['attribute']]
        scalar=torch.tensor(value,dtype=torch.float64)
        legacy.append(dict(source=row['source'],attributes=[],
            spec=dict(shape=list(scalar.shape),dtype=str(scalar.dtype),stride=list(scalar.stride())),
            bytes=scalar.element_size(),sha256=hashlib.sha256(
                bytes(scalar.reshape(1).view(torch.uint8).numpy())).hexdigest(),
            kind='immutable_Qwen3_module_scalar'))
    expected=binding_content_signature(tensors+legacy)
    assert selected_model_content_signature(tensors,rows)==expected
    changed=copy.deepcopy(rows);raw=bytes.fromhex('000000000000d03f')
    changed[0]['hex_le']=raw.hex();changed[0]['sha256']=hashlib.sha256(raw).hexdigest()
    assert selected_model_content_signature(tensors,changed)!=expected
    changed_tensors=copy.deepcopy(tensors);changed_tensors[0]['sha256']='0'*64
    assert selected_model_content_signature(changed_tensors,rows)!=expected
    with pytest.raises(ValueError,match='tensor bindings'):
        selected_model_content_signature(tensors[:-1],rows)


@pytest.mark.parametrize('fault',['missing','duplicate','source','module','bytes','digest','nan','order'])
def test_incomplete_or_corrupt_constant_contract_rejected(fault):
    rows=GuardedModuleConstants(model()).records
    if fault=='missing':rows.pop()
    elif fault=='duplicate':rows[0]=copy.deepcopy(rows[1])
    elif fault=='source':rows[0]['source']='wrong'
    elif fault=='module':rows[0]['module']='model.__dict__'
    elif fault=='bytes':rows[0]['bytes']=True
    elif fault=='digest':rows[0]['sha256']='0'*64
    elif fault=='nan':
        rows[0]['hex_le']='000000000000f87f';rows[0]['sha256']=hashlib.sha256(bytes.fromhex(rows[0]['hex_le'])).hexdigest()
    else:rows.reverse()
    with pytest.raises(ValueError):validate_constant_rows(rows)


def declaration():
    rows=GuardedModuleConstants(model()).records
    return dict(immutable_inputs=[{} for _ in range(509)],
        named_bindings=[{'aliases':[{}]} for _ in range(509)],
        guarded_module_constants=dict(schema=SCHEMA,
            source_sha256=hashlib.sha256(Path(constants.__file__).read_bytes()).hexdigest(),
            records=rows,sha256=constant_signature(rows)))


def test_new_capture_convention_cannot_drop_checks_or_mix_legacy_scalars():
    assert validate_capture_constants({})==651
    record=declaration();assert validate_capture_constants(record)==509
    for fault in ['source','count','mixed','digest','mode']:
        record=declaration()
        if fault=='source':record['guarded_module_constants']['source_sha256']='0'*64
        elif fault=='count':record['immutable_inputs'].pop()
        elif fault=='mixed':record['immutable_inputs'][0]['kind']='immutable_Qwen3_module_scalar'
        elif fault=='digest':record['guarded_module_constants']['sha256']='0'*64
        else:record['guarded_module_constants']['schema']='unsupported'
        with pytest.raises(ValueError):validate_capture_constants(record)


def test_compiler_binding_validates_constants_before_tensor_reads(monkeypatch):
    from hexagon_carrier_parameter_binding import CarrierParameterBinding
    m=model();record=declaration();record['graph_identity']={'inputs':510}
    m.model.layers[2].self_attn.scaling=0.25
    monkeypatch.setattr(m,'get_parameter',lambda *unused:pytest.fail('Changed constants must reject before tensor reads'))
    with pytest.raises(ValueError,match='constant bytes differ'):CarrierParameterBinding(m,record)


def test_constant_corruption_rejected_before_graph_unpickle(tmp_path,monkeypatch):
    import json
    from torch.fx._graph_pickler import GraphPickler
    from restore_qwen_symbolic_graph_carrier import restore_carrier
    from hexagon_live_compile_mailbox import LIVE_STATUS,CONTEXT_SHA,GENERATION_SHA,POLICY_SHA
    import hexagon_live_compile_mailbox as live
    record=declaration()
    record.update(status=LIVE_STATUS,source_sha256=hashlib.sha256(Path(live.__file__).read_bytes()).hexdigest(),
        host_machine='aarch64',prior_session_DSP_calls=0,request_graph_executed=False,
        CPU_model_forward=False,context_sha256=CONTEXT_SHA,generation_sha256=GENERATION_SHA,
        decomposition_sha256=POLICY_SHA,torch_version=torch.__version__,torch_git=torch.version.git_version)
    record['guarded_module_constants']['records'].pop()
    path=tmp_path/'result.json';path.write_text(json.dumps(record))
    monkeypatch.setattr(GraphPickler,'loads',lambda *unused:pytest.fail('Must reject before graph unpickle'))
    with pytest.raises(ValueError,match='source set'):restore_carrier(path,hashlib.sha256(path.read_bytes()).hexdigest(),trusted_local=True)
