"""Read-only named-module hooks for exact-packet numerical diagnosis.

Uses PyTorch hooks; it does not replace model operations or generation/cache
logic. Callers must compare the normal outputs to an unobserved baseline.
"""
import hashlib
from pathlib import Path
import re


class NativeModuleObservation:
    def __init__(self, model, names, output):
        if len(set(names)) != len(names):
            raise ValueError('Unique observation modules required')
        self.targets = []
        for name in names:
            if (not isinstance(name, str) or not re.fullmatch(r'[A-Za-z_][A-Za-z0-9_.]*', name)
                    or any(not part for part in name.split('.'))):
                raise ValueError('Explicit dotted module name required')
            self.targets.append((name, model.get_submodule(name)))
        self.output = Path(output)
        self.rows = []
        self.handles = []
        self.seen = set()

    def capture(self, name, role, value):
        import torch
        if (not isinstance(value, torch.Tensor) or value.dtype != torch.float32
                or not bool(torch.isfinite(value).all())):
            raise ValueError('Finite FP32 native module observation required')
        key = (name, role)
        if key in self.seen:
            raise ValueError('Module ran more than once in exact forward: ' + name)
        self.seen.add(key)
        payload = value.detach().contiguous().cpu().numpy().tobytes()
        path = self.output/(name+'-'+role+'.f32.bin')
        with path.open('xb') as stream:
            stream.write(payload)
        self.rows.append(dict(module=name, role=role, shape=list(value.shape),
                              path=str(path), bytes=len(payload),
                              sha256=hashlib.sha256(payload).hexdigest()))

    def __enter__(self):
        try:
            for name, module in self.targets:
                def before(module, positional, name=name):
                    if len(positional) != 1:
                        raise ValueError('Expected one tensor module input: ' + name)
                    self.capture(name, 'input', positional[0])

                def after(module, positional, output, name=name):
                    self.capture(name, 'output', output)

                self.handles.append(module.register_forward_pre_hook(before))
                self.handles.append(module.register_forward_hook(after))
        except BaseException:
            self.close()
            raise
        return self

    def close(self):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()

    def __exit__(self, exc_type, exc, tb):
        self.close()
        if exc_type is None and len(self.rows) != 2 * len(self.targets):
            raise ValueError('Not all requested native modules ran')
