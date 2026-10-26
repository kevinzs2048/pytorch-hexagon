"""Admitted Qwen query bounds, independent of prompt contents or capture hints."""

MAX_QUERY = 32
KV_CAPACITY = 1024


def query_bounds(extent):
    """Intersect authoritative captured ranges with this backend's resource bound."""
    import sympy
    import torch

    expression = extent.node.expr if isinstance(extent, torch.SymInt) else extent
    if type(expression) is int or isinstance(expression, sympy.Integer):
        value = int(expression)
        if not 1 <= value <= MAX_QUERY:
            raise ValueError('Captured query is outside the admitted execution budget')
        return value, value
    if not isinstance(expression, sympy.Symbol):
        raise ValueError('Query requires a literal or one captured shape symbol')
    limits = extent.node.shape_env.var_to_range.get(expression)
    if limits is None:
        raise ValueError('Captured query symbolic range missing')
    lower = max(sympy.Integer(1), limits.lower)
    upper = min(sympy.Integer(MAX_QUERY), limits.upper)
    if lower > upper or not lower.is_Integer or not upper.is_Integer:
        raise ValueError('Captured query range does not intersect the execution budget')
    return int(lower), int(upper)


def contract_query_bounds(contract):
    """Validate the shared resource policy before generating or compiling a caller."""
    variables = contract.get('variables')
    if not isinstance(variables, dict) or set(variables) != {'query'}:
        raise ValueError('One declared Qwen query variable required')
    bounds = variables['query']
    if (not isinstance(bounds, (list, tuple)) or len(bounds) != 2
            or any(type(value) is not int for value in bounds)
            or not 1 <= bounds[0] <= bounds[1] <= MAX_QUERY):
        raise ValueError('Query contract exceeds the admitted execution budget')
    return tuple(bounds)


def annotate_query_inputs(kwargs):
    """Mark coherent query axes before Dynamo, leaving native forward data intact."""
    import torch

    ids = kwargs.get('input_ids')
    if (not isinstance(ids, torch.Tensor) or ids.dtype != torch.int64
            or ids.ndim != 2 or ids.shape[0] != 1):
        raise ValueError('Batch-one int64 input_ids required at compiled forward boundary')
    query = ids.shape[1]
    query_bounds(query)
    axes = [(ids, 1)]
    for name, rank in (('cache_position', 1), ('position_ids', 2)):
        value = kwargs.get(name)
        if value is None:
            continue
        if (not isinstance(value, torch.Tensor) or value.dtype != torch.int64
                or value.ndim != rank or value.shape[-1] != query
                or (rank == 2 and value.shape[0] != 1)):
            raise ValueError('Native position axes must agree with query')
        axes.append((value, rank-1))
    mask_argument = kwargs.get('attention_mask')
    masks = mask_argument.values() if isinstance(mask_argument, dict) else (mask_argument,)
    for mask in masks:
        if mask is None:
            continue
        if not isinstance(mask, torch.Tensor) or mask.shape[0] != 1:
            raise ValueError('Native batch-one attention mask required')
        if mask.ndim == 2:
            if not query <= mask.shape[1] <= KV_CAPACITY:
                raise ValueError('Native attention mask exceeds cache capacity or omits query')
            if mask.shape[1] > 1:
                torch._dynamo.mark_dynamic(mask, 1, min=2, max=KV_CAPACITY)
        elif mask.ndim == 4:
            if mask.shape[2] != query or mask.shape[3] != KV_CAPACITY:
                raise ValueError('Native causal mask must match query and cache capacity')
            axes.append((mask, 2))
        else:
            raise ValueError('Supported native attention mask rank required')
    if query > 1:
        for value, axis in axes:
            torch._dynamo.mark_dynamic(value, axis, min=2, max=MAX_QUERY)


def compiled_query_scope(model):
    """Annotate inputs around HF's compiled call without replacing forward/generate.

    Native generate still selects and obtains the cached torch.compile callable.
    Annotation runs outside that callable because mark_dynamic cannot be traced.
    Restore the getter even when capture or device execution raises.
    """
    from contextlib import contextmanager
    from functools import wraps

    @contextmanager
    def scope():
        original = model.get_compiled_call
        missing = object()
        previous = model.__dict__.get('get_compiled_call', missing)

        @wraps(original)
        def get_compiled_call(config):
            compiled = original(config)

            @wraps(compiled)
            def call(*args, **kwargs):
                if args:
                    raise ValueError('Native compiled forward keyword inputs required')
                annotate_query_inputs(kwargs)
                return compiled(**kwargs)

            return call

        model.get_compiled_call = get_compiled_call
        try:
            yield
        finally:
            if previous is missing:
                delattr(model, 'get_compiled_call')
            else:
                model.get_compiled_call = previous

    return scope()
