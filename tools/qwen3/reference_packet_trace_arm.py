"""Independent native W8A8 actual-input/output trace with optional exact snapshots."""
import argparse,hashlib,json,platform,sys,traceback
from pathlib import Path
from packet_trace import describe_packet,actual_native_inputs,actual_native_outputs,validate_qwen_layout


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('sources','context','hf','plan','output'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--plan-sha256',required=True)
    p.add_argument('--snapshot-step',type=int)
    a=p.parse_args()
    if platform.machine()!='aarch64' or hashlib.sha256(a.plan.read_bytes()).hexdigest()!=a.plan_sha256:
        raise ValueError('Actual ARM and immutable generation/ABI plan required')
    sys.path[:0]=[str(a.sources/'qwen3/runtime/arm'),str(a.sources)]
    import torch
    from run_qwen_arm_dynamic_warm_generate import load_model,tokenize_prompts,terminal_observation,validate_requests
    from hexagon_native1024_cache_setup import prepare_native1024_cache
    from qcom_hexagon_backend.backend.hexagon_dynamic_buffer_io import layout_for
    from hexagon_sparse_packet_storage import sparsify_verified_packet
    from hexagon_live_compile_mailbox import atomic_json
    torch.set_num_threads(2)
    plan=json.loads(a.plan.read_text());a.output.mkdir(exist_ok=False)
    r=dict(schema=1,status='independent_native_trace_running',CPU_reference_execution=True,
        CPU_model_fallback=False,DSP_calls=0,plan_sha256=a.plan_sha256,
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        codec_sha256=hashlib.sha256(Path(__file__).with_name('packet_trace.py').read_bytes()).hexdigest(),
        context_sha256=hashlib.sha256(a.context.read_bytes()).hexdigest(),
        generation_config_sha256=hashlib.sha256((a.hf/'generation_config.json').read_bytes()).hexdigest(),
        torch_version=torch.__version__,torch_git=torch.version.git_version,forwards=[],requests=[])
    def save():atomic_json(a.output/'result.json',r)
    save()
    class SnapshotComplete(Exception):pass
    try:
        if 'prompts' in plan:
            _,requests,provenance=tokenize_prompts(a.hf,plan['prompts'],chat=plan.get('chat_template',False))
            r['input_source']=provenance
        else:requests=plan['requests'];r['input_source']={'kind':'explicit_token_ids'}
        validate_requests(requests,plan['max_new_tokens'],plan['chunk_size'])
        model=load_model(a.context,a.hf);cache=prepare_native1024_cache(model)
        eos=model.generation_config.eos_token_id;eos=[] if eos is None else eos if isinstance(eos,list) else [eos]
        r['generation_options']=dict(max_new_tokens=plan['max_new_tokens'],prefill_chunk_size=plan['chunk_size'],native_eos_token_ids=eos,do_sample=False,native_cache_capacity=1024)
        pending=None;request_index=0
        def contract(q):
            found=[c for c in plan['contracts'] if c['variables']['query'][0]<=q<=c['variables']['query'][1]]
            if len(found)!=1:raise ValueError('One verified ABI for the actual native query required')
            return found[0]
        def before(mod,args,kwargs):
            nonlocal pending
            q=kwargs['input_ids'].shape[1];step=len(r['forwards']);c=contract(q);layout=layout_for(c,{'query':q});validate_qwen_layout(layout,q)
            pending=dict(step=step,request_index=request_index,query=q,
                token_ids=kwargs['input_ids'].reshape(-1).tolist(),positions=kwargs['cache_position'].reshape(-1).tolist(),contract=c)
            file=a.output/f'input-{step:04d}.bin' if step==a.snapshot_step else None
            pending['input']=describe_packet(layout,'inputs',actual_native_inputs(kwargs,cache),query=q,save=file)
            if file is not None:pending['input']['snapshot']=str(file);sparsify_verified_packet(file,pending['input']['sha256'])
        def after(mod,args,kwargs,out):
            layout=layout_for(pending['contract'],{'query':pending['query']});step=pending['step']
            file=a.output/f'output-{step:04d}.bin' if step==a.snapshot_step else None
            pending['output']=describe_packet(layout,'outputs',actual_native_outputs(out,cache),save=file)
            if file is not None:pending['output']['snapshot']=str(file);sparsify_verified_packet(file,pending['output']['sha256'])
            r['forwards'].append(pending);save();print('native reference',step,'query',pending['query'],flush=True)
            if step==a.snapshot_step:raise SnapshotComplete()
        for request_index,prompt in enumerate(requests):
            first=len(r['forwards']);ids=torch.tensor([prompt],dtype=torch.int64)
            pre=model.register_forward_pre_hook(before,with_kwargs=True);post=model.register_forward_hook(after,with_kwargs=True)
            try:
                with torch.no_grad():result=model.generate(ids,attention_mask=torch.ones_like(ids),
                    cache_implementation='static',disable_compile=True,do_sample=False,
                    max_new_tokens=plan['max_new_tokens'],prefill_chunk_size=plan['chunk_size'],
                    return_dict_in_generate=True,output_logits=True)
            finally:pre.remove();post.remove()
            tokens=result.sequences[0,len(prompt):].tolist()
            if model._cache is not cache or cache.max_cache_len!=1024:raise ValueError('Native reference cache identity/capacity differs')
            r['requests'].append(dict(index=request_index,prompt_ids=prompt,generated_ids=tokens,
                first_forward=first,end_forward=len(r['forwards']),terminal_observation=terminal_observation(tokens,plan['max_new_tokens'],eos)))
            save()
        r.update(status='independent_native_packet_trace_completed',full_native_generate=True)
    except SnapshotComplete:
        r.update(status='independent_native_exact_snapshot_recorded',full_native_generate=False)
    except BaseException:
        r.update(status='failed',error=traceback.format_exc());raise
    finally:save()
    print(r['status'],flush=True)


if __name__=='__main__':main()
