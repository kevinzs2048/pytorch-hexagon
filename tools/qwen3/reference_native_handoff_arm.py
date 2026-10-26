"""Read-only ARM W8A8 controls: exact DSP input replay and native generate trajectory."""
import argparse,hashlib,json,platform,sys,traceback
from pathlib import Path
p=argparse.ArgumentParser()
for n in ('sources','run','context','hf','output'):p.add_argument('--'+n,type=Path,required=True)
p.add_argument('--run-sha256',required=True)
a=p.parse_args();sys.path.insert(0,str(a.sources/'qwen3/runtime/arm'));sys.path.insert(0,str(a.sources))
import numpy as np
import torch
from run_qwen_arm_dynamic_warm_generate import load_model,terminal_observation
from reference_qwen_dynamic_exact_packet_arm import unpack_request
from hexagon_native1024_cache_setup import prepare_native1024_cache
from hexagon_compiled_artifact_cache import digest_file,canonical
from audit_qwen_native_request_sequence import audit_unchanged_kv,kv_packet_digests,audit_cache_handoff,audit_logits
from hexagon_library_lifecycle import audit_library_events
from qwen_native_request_sequence import verify_request
from qcom_hexagon_backend.backend.hexagon_dynamic_buffer_io import layout_for
from audit_saved_packet_library_comparison import tensor_difference
assert platform.machine()=='aarch64' and digest_file(a.run)==a.run_sha256
run=json.loads(a.run.read_text())
assert run['status']=='ARM_native_dynamic_compiler_handoff_generate_completed' and run['process_exit_code']==0
assert run['complete_raw_packets_retained'] and not run['CPU_model_fallback']
assert len(run['forwards'])==run['DSP_calls']
a.output.mkdir()
r=dict(status='running_ARM_native_numerical_controls',source_sha256=digest_file(__file__),run_sha256=a.run_sha256,context_sha256=digest_file(a.context),DSP_calls=0,CPU_reference_execution=True,numerical_acceptance=False,same_input=[],native_requests=[])
def save():(a.output/'result.json').write_bytes(canonical(r)+b'\n')
save()
def array(t):return t.detach().contiguous().numpy()
def metric(x,y):return tensor_difference(np.asarray(x,dtype=np.float32),np.asarray(y,dtype=np.float32))
def dump(folder,name,t):
 path=folder/name;path.write_bytes(np.ascontiguousarray(t).tobytes());return dict(path=str(path),sha256=digest_file(path),shape=list(t.shape))
def read_dsp(row):
 folder=a.run.parent/f"forward-{row['step']:02d}-q{row['query']}"
 inc=(folder/'input.bin').read_bytes();out=(folder/'output.bin').read_bytes()
 assert hashlib.sha256(inc).hexdigest()==row['input_sha256'] and hashlib.sha256(out).hexdigest()==row['output_sha256']
 contract=run['captures'][row['AOT_capture_index']]['contract'];layout=layout_for(contract,{'query':row['query']})
 assert len(out)==layout['output_bytes'] and len(inc)==16+layout['input_bytes']
 vals=[np.frombuffer(out,dtype='<f4',count=s['bytes']//4,offset=s['offset']).reshape(s['shape']) for s in layout['outputs']]
 assert len(vals)==57 and all(np.isfinite(v).all() for v in vals)
 scope=audit_unchanged_kv(inc,out,layout,row['cache_positions'])
 return vals,scope
try:
 # Validate full native KV handoff and reset from retained raw bytes before CPU execution.
 r['packet_audit']=[]
 r['library_lifecycle']=audit_library_events(run['stream_events'],[f['DSP_library_slot'] for f in run['forwards']])
 previous=None
 for row in run['forwards']:
  folder=a.run.parent/f"forward-{row['step']:02d}-q{row['query']}"
  incoming=(folder/'input.bin').read_bytes();outgoing=(folder/'output.bin').read_bytes()
  assert hashlib.sha256(incoming).hexdigest()==row['input_sha256'] and hashlib.sha256(outgoing).hexdigest()==row['output_sha256']
  layout=layout_for(run['captures'][row['AOT_capture_index']]['contract'],{'query':row['query']})
  scope=audit_unchanged_kv(incoming,outgoing,layout,row['cache_positions'])
  if row['cache_positions'][0]==0:
   for field in layout['inputs'][3:5]+layout['inputs'][6:]:
    view=np.frombuffer(incoming,dtype='<u4',count=field['bytes']//4,offset=16+field['offset']);assert not np.any(view)
   continuity={'reset_KV_all_zero':True}
  else:continuity=audit_cache_handoff(previous,incoming,layout)
  previous=kv_packet_digests(outgoing,layout,side='output')
  r['packet_audit'].append(dict(step=row['step'],scope=scope,continuity=continuity,logits=audit_logits(outgoing,layout,row['logits_top_token'])))
 del incoming,outgoing
 save()
 torch.set_num_threads(2)
 r['torch_version']=torch.__version__;r['torch_git']=torch.version.git_version;r['torch_threads']=torch.get_num_threads()
 model=load_model(a.context,a.hf);cache=prepare_native1024_cache(model)
 kv=[v for l in cache.layers for v in (l.keys,l.values)]
 for request in run['requests']:
  verify_request(run['forwards'][request['first_forward']:request['end_forward']],request['prompt_ids'],request['generated_ids'])
 for row in run['forwards']:
  folder=a.run.parent/f"forward-{row['step']:02d}-q{row['query']}";q=row['query'];start=row['cache_positions'][0]
  ts=unpack_request(folder/'input.bin',q,start); assert ts[0].reshape(-1).tolist()==row['input_token_ids']
  ds,scope=read_dsp(row)
  with torch.no_grad():
   for dst,src in zip(kv,ts[3:5]+ts[6:],strict=True):dst.copy_(src)
  assert all(torch.equal(dst,src) for dst,src in zip(kv,ts[3:5]+ts[6:],strict=True))
  with torch.inference_mode():out=model.forward(input_ids=ts[0],cache_position=ts[1],position_ids=ts[2],past_key_values=cache,attention_mask={'full_attention':ts[5]},use_cache=True,return_dict=True)
  assert out.past_key_values is cache
  logits=array(out.logits[:,-1:,:]);updated=np.stack([array(v[:,:,start:start+q,:]) for v in kv])
  dest=a.output/f"same-input-{row['step']:02d}";dest.mkdir()
  kmetrics=[metric(ds[i][:,:,start:start+q,:],updated[i]) for i in range(56)]
  item=dict(step=row['step'],query=q,positions=row['cache_positions'],input_sha256=row['input_sha256'],output_sha256=row['output_sha256'],inputs_exact=True,KV_update_scope=scope,logits=metric(ds[-1],logits),updated_KV=kmetrics,reference_logits=dump(dest,'logits.bin',logits),reference_updated_KV=dump(dest,'updated-kv.bin',updated))
  r['same_input'].append(item);save();print('same input',row['step'],item['logits'],flush=True)
 # Fresh native cache; generate() owns token selection, chunking and stopping.
 cache.reset()
 options=run['generation_options']
 for request in run['requests']:
  events=[];step_start=request['first_forward'];group=run['forwards'][step_start:request['end_forward']]
  dest=a.output/f"native-request-{request['index']:02d}";dest.mkdir()
  def observe(mod,pos,kwargs,out):
   ids=kwargs['input_ids'].reshape(-1).tolist();positions=kwargs['cache_position'].reshape(-1).tolist();valid=max(positions)+1
   logits=array(out.logits[:,-1:,:]);valid_kv=np.stack([array(v[:,:,:valid,:]) for v in kv]);idx=len(events)
   item=dict(index=idx,token_ids=ids,positions=positions,reference_logits=dump(dest,f'{idx:02d}-logits.bin',logits),reference_valid_KV=dump(dest,f'{idx:02d}-valid-kv.bin',valid_kv))
   if idx<len(group):
    row=group[idx];ds,_=read_dsp(row);item['same_positions']=row['cache_positions']==positions;item['same_token_ids']=row['input_token_ids']==ids
    if item['same_positions']:
     item['logits']=metric(ds[-1],logits);item['valid_KV']=[metric(ds[i][:,:,:valid,:],valid_kv[i]) for i in range(56)]
   events.append(item)
  hook=model.register_forward_hook(observe,with_kwargs=True)
  try:
   ids=torch.tensor([request['prompt_ids']],dtype=torch.int64)
   with torch.no_grad():generated=model.generate(ids,attention_mask=torch.ones_like(ids),cache_implementation='static',disable_compile=True,do_sample=False,max_new_tokens=options['max_new_tokens'],prefill_chunk_size=options['prefill_chunk_size'],return_dict_in_generate=True,output_logits=True)
  finally:hook.remove()
  tokens=generated.sequences[0,len(request['prompt_ids']):].tolist()
  terminal=terminal_observation(tokens,options['max_new_tokens'],options['native_eos_token_ids'])
  item=dict(index=request['index'],generated_ids=tokens,same_generated_ids=tokens==request['generated_ids'],terminal_observation=terminal,same_terminal=terminal==request['terminal_observation'],same_forward_count=len(events)==len(group),forwards=events)
  assert model._cache is cache
  r['native_requests'].append(item);save();print('native',request['index'],tokens,'same',item['same_generated_ids'],flush=True)
 r['status']='ARM_native_same_input_and_generate_numerics_recorded_NOT_acceptance';save()
except BaseException:
 r.update(status='failed',error=traceback.format_exc());save();raise
