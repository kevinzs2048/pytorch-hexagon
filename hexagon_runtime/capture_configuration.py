"""Scoped native Dynamo configuration; tensor/query inputs stay dynamic."""


def capture_configuration(*, specialize_floats=False):
    if type(specialize_floats) is not bool:
        raise ValueError('Explicit boolean float specialization required')
    import torch
    options=dict(suppress_errors=False)
    if specialize_floats:
        options['specialize_float']=True
    return torch._dynamo.config.patch(**options)
