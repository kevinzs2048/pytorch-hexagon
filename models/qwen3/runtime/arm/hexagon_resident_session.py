"""Internal backend for a verified resident-session bundle.

Generation/cache semantics stay with Transformers. Each callback validates its
captured graph, invokes the DSP and reconstructs native cache outputs. Optional
snapshots and observers are diagnostics; normal calls retain metadata only.
"""
import copy, json, re, time
from pathlib import Path
from hexagon_dynamic_cache_graph import canonical, role
import torch
import torch._dynamo.backends.common as aot_common
from functorch.compile import make_boxed_func
from hexagon_dynamic_module_constants import DynamicParameterBinding
from hexagon_module_scalar_binding import bind_module_scalar_attributes
from hexagon_dynamo_parameters import resolve_descriptor
from hexagon_native_silu import preserve_native_silu
from hexagon_runtime_decompositions import load_runtime_decompositions
from models.qwen3.runtime.support.exact_parameter_bytes import ExactParameterBytePool
from models.qwen3.runtime.arm import hexagon_resident_protocol as protocol

class ResidentSessionBackend:

    def __init__(self, model, stream, root, out, *, verify_cache=False, observe=None, diagnostics=False):
        self.model, self.stream, self.root, self.out = (model, stream, root, out)
        self.verify_cache, self.observe = (verify_cache, observe)
        self.diagnostics = diagnostics
        self.calls = []
        self.captures = []
        self.parameter_pool = ExactParameterBytePool(workers=4)
        self.position = 0
        self.identities = {}
        for name in ('initial', 'prefill', 'decode', 'single0', 'single1'):
            sources = [x['source'] for x in json.loads((root / (name + '-mapping.json')).read_text())['inputs']]
            self.identities[name] = canonical((root / (name + '-graph.txt')).read_text(), sources)
        decode_sources = [x['source'] for x in json.loads((root / 'decode-mapping.json').read_text())['inputs']]
        self.single1_identity = canonical((root / 'decode-graph.txt').read_text(), decode_sources, known={'past': 1}, scalar_literals=['_tensor_constant0'])
        single_sources = [x['source'] for x in json.loads((root / 'single1-mapping.json').read_text())['inputs']]
        if not (self.single1_identity == canonical((root / 'single1-graph.txt').read_text(), single_sources, scalar_literals=['_tensor_constant0'])):
            raise ValueError("Resident input/output contract failed: self.single1_identity == canonical((root / 'single1-graph.txt').read_text(), single_sources, scalar_literals=['_tensor_constant0'])")
        self.table = preserve_native_silu(load_runtime_decompositions(root / 'decompositions.py', expected_sha256='2c1583eb71ef9a3f7791d312f2cef7badb413068950134a39729f19b2b7f48e5', expected_torch_version='2.12.0+cpu', expected_torch_git='7661cd9c6b841b62b7f411aa52ec51f05457263b'))

    def header(self, mode, position, query=0):
        return protocol.header(mode, position, query)

    def reset(self):
        self.stream.invoke_packet(self.header(0, 0), 64)
        self.position = 0

    def __call__(self, dg, di):

        def compiler(gm, examples):
            binding = DynamicParameterBinding(dg, di, gm, examples, input_info=torch._guards.TracingContext.get().fw_metadata.input_info)
            bind_module_scalar_attributes(binding, dg, di, gm, examples, input_info=torch._guards.TracingContext.get().fw_metadata.input_info, roots={'self': self.model}, allowed_attributes={'variance_epsilon', 'scaling', 'attention_scaling'})
            private = torch.fx.GraphModule(gm, copy.deepcopy(gm.graph))
            binding.freeze(private)
            dn = [n for n in dg.graph.nodes if n.op == 'placeholder']
            an = [n for n in gm.graph.nodes if n.op == 'placeholder']
            sources = []
            for j in binding.retained:
                k, _, attrs = resolve_descriptor(an[j].meta['desc'], di)
                if not (not attrs):
                    raise ValueError('Resident input/output contract failed: not attrs')
                sources.append(dn[k].meta['grapharg'].source.name.replace("L['kwargs'][", 'L['))
            identity = canonical(private.code, sources)
            matches = [name for name, value in self.identities.items() if identity == value]
            if len(matches) > 1:
                raise RuntimeError('Captured graph matches multiple compiled domains')
            kind = matches[0] if matches else None
            if kind is None:
                (self.out / f'unsupported-graph-{len(self.captures)}.txt').write_text(private.code)
                (self.out / f'unsupported-sources-{len(self.captures)}.json').write_text(json.dumps(sources, indent=2))
                raise RuntimeError('Native graph does not exactly match a compiled domain')
            if kind == 'single1':
                literal = getattr(private, '_tensor_constant0')
                from torch.utils._python_dispatch import _disable_current_modes
                from torch._subclasses.fake_tensor import is_fake
                if not (not is_fake(literal) and literal.ndim == 0 and (literal.dtype == torch.float32)):
                    raise ValueError('Resident input/output contract failed: not is_fake(literal) and literal.ndim == 0 and (literal.dtype == torch.float32)')
                with _disable_current_modes():
                    if not (literal.item() == 0.0):
                        raise ValueError('Resident input/output contract failed: literal.item() == 0.0')
                if not (canonical(private.code, sources, scalar_literals=['_tensor_constant0']) == self.single1_identity):
                    raise ValueError("Resident input/output contract failed: canonical(private.code, sources, scalar_literals=['_tensor_constant0']) == self.single1_identity")
            idx = sources.index("L['input_ids']")
            cp = sources.index("L['cache_position']")
            pi = sources.index("L['position_ids']")
            am = sources.index("L['attention_mask']")
            planes = {}
            for i, source in enumerate(sources):
                m = re.fullmatch("L\\['past_key_values'\\]\\.layers\\[(\\d+)\\]\\.(keys|values)", source)
                if m:
                    planes[2 * int(m[1]) + (m[2] == 'values')] = i
            if not ((len(planes) == 56) == (kind not in ('initial', 'single0'))):
                raise ValueError("Resident input/output contract failed: (len(planes) == 56) == (kind not in ('initial', 'single0'))")
            capture = len(self.captures)
            self.captures.append({'kind': kind, 'sources': sources, 'frozen': len(binding.records)})
            parameter_guard = self.parameter_pool.bind(binding)
            # The guard owns pinned metadata and shares immutable byte snapshots
            # across captures. The private matching graph will not execute.
            binding.values.clear()

            def run(*values):
                started = time.perf_counter()
                bound = parameter_guard.check(values)
                ids = bound[idx].contiguous()
                q = ids.shape[1]
                position = self.position
                if not (ids.shape[0] == 1 and 1 <= q <= 32 and (position + q <= 1024)):
                    raise ValueError('Resident input/output contract failed: ids.shape[0] == 1 and 1 <= q <= 32 and (position + q <= 1024)')
                allowed = {'initial': position == 0 and q >= 2, 'single0': position == 0 and q == 1, 'single1': position == 1 and q == 1, 'decode': position >= 2 and q == 1, 'prefill': position > 0 and q >= 2}
                if not (allowed[kind]):
                    raise ValueError((kind, position, q))
                if not (ids.dtype == torch.int64 and all((bound[i].dtype == torch.int64 for i in (cp, pi, am)))):
                    raise ValueError('Resident input/output contract failed: ids.dtype == torch.int64 and all((bound[i].dtype == torch.int64 for i in (cp, pi, am)))')
                if not (torch.equal(bound[cp], torch.arange(position, position + q))):
                    raise ValueError('Resident input/output contract failed: torch.equal(bound[cp], torch.arange(position, position + q))')
                if not (torch.equal(bound[pi], torch.arange(position, position + q).unsqueeze(0))):
                    raise ValueError('Resident input/output contract failed: torch.equal(bound[pi], torch.arange(position, position + q).unsqueeze(0))')
                if not (bound[am].shape == (1, position + q) and bool((bound[am] == 1).all())):
                    raise ValueError('Resident input/output contract failed: bound[am].shape == (1, position + q) and bool((bound[am] == 1).all())')
                for i, source in enumerate(sources):
                    semantic = role(source)
                    if semantic in ('query', 'past', 'total'):
                        if not (int(bound[i]) == {'query': q, 'past': position, 'total': position + q}[semantic]):
                            raise ValueError(source)
                    if source == "L['logits_to_keep']":
                        if not (int(bound[i]) == 1):
                            raise ValueError('Resident input/output contract failed: int(bound[i]) == 1')
                for p, i in planes.items():
                    if not (bound[i].dtype == torch.float32 and bound[i].shape == (1, 8, position, 128)):
                        raise ValueError((p, bound[i].shape, position))
                checked = time.perf_counter()
                logits, delta, stats, event, exchanged = protocol.forward(self.stream, ids, position)
                full = [torch.cat((bound[planes[p]], delta[p]), dim=2) if kind not in ('initial', 'single0') else delta[p].contiguous() for p in range(56)]
                reconstructed = time.perf_counter()
                if self.verify_cache:
                    protocol.verify_cache(self.stream, position + q, full)
                self.position += q
                row = {'capture': capture, 'position': position, 'query': q, 'ids': ids.tolist(), 'statistics': stats, 'invoke': event, 'parameter_check_s': checked - started, 'exchange_s': exchanged - checked, 'reconstruct_s': reconstructed - exchanged, 'snapshot_check_s': time.perf_counter() - reconstructed}
                if self.diagnostics:
                    row['top_token'] = int(logits.reshape(-1).argmax())
                self.calls.append(row)
                if self.observe is not None:
                    self.observe(logits, full, row)
                return tuple((full[p ^ 1] for p in range(56))) + (logits,)
            return make_boxed_func(run)

        def no_backward(*unused):
            raise RuntimeError('Inference only')
        return aot_common.aot_autograd(fw_compiler=compiler, inference_compiler=compiler, bw_compiler=no_backward, decompositions=self.table)(dg, di)
