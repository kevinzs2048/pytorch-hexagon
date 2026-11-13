"""Compare the captured Qwen graph under validated cache-shape equalities.

canonical inlines integer shape arithmetic and removes aliases/proven full-span slices. Floating-point
operations stay ordered. Callers must check every source-role equality and any
scalar-literal shapes before reusing the compiled entry.
"""
import ast
import copy
import re

def role(source):
    source = source.replace("L['kwargs'][", 'L[')
    if source in ("L['input_ids'].size()[1]", "L['cache_position'].size()[0]"):
        return 'query'
    if source == "L['attention_mask'].size()[1]":
        return 'total'
    if re.fullmatch("L\\['past_key_values'\\]\\.layers\\[\\d+\\]\\.(keys|values)\\.size\\(\\)\\[2\\]", source):
        return 'past'
    return source

def canonical(source, sources, known=None, scalar_literals=()):
    known = known or {}
    function = ast.parse(source).body[0]
    args = [a.arg for a in function.args.args if a.arg != 'self']
    if len(args) != len(sources):
        raise ValueError("Graph source count differs from argument count")
    replacements = {a: ast.Constant(value=known[role(s)]) if role(s) in known else ast.Name(id=role(s), ctx=ast.Load()) for a, s in zip(args, sources)}
    integer_names = {'query', 'total', 'past'}
    rows = []
    shapes = {}

    class Replace(ast.NodeTransformer):

        def visit_BinOp(self, node):
            node = self.generic_visit(node)
            if isinstance(node.left, ast.Constant) and type(node.left.value) is int and isinstance(node.right, ast.Constant) and (type(node.right.value) is int):
                a, b = (node.left.value, node.right.value)
                if isinstance(node.op, ast.Add):
                    return ast.Constant(value=a + b)
                if isinstance(node.op, ast.Sub):
                    return ast.Constant(value=a - b)
                if isinstance(node.op, ast.Mult):
                    return ast.Constant(value=a * b)
            return node

        def visit_Name(self, node):
            replacement = replacements.get(node.id)
            if replacement is None:
                return node
            if isinstance(replacement, ast.Name):
                return ast.Name(id=replacement.id, ctx=ast.Load())
            if isinstance(replacement, ast.Constant):
                return ast.Constant(value=replacement.value)
            return copy.deepcopy(replacement)

    def integer(node):
        if isinstance(node, ast.Name):
            return node.id in integer_names
        if isinstance(node, ast.Constant):
            return type(node.value) is int
        if isinstance(node, ast.BinOp):
            return isinstance(node.op, (ast.Add, ast.Sub, ast.Mult)) and integer(node.left) and integer(node.right)
        return False
    for statement in function.body:
        if isinstance(statement, ast.Assign):
            if isinstance(statement.value, ast.Constant) and statement.value.value is None:
                continue
            if len(statement.targets) != 1 or not isinstance(statement.targets[0], ast.Name):
                raise ValueError("Only single-name FX assignments are supported")
            name = statement.targets[0].id
            # ast.parse created a private tree for this call. Only replacement
            # expressions shared across statements need copying.
            value = Replace().visit(statement.value)
            if isinstance(value, ast.Call) and isinstance(value.func, ast.Attribute) and isinstance(value.func.value, ast.Name) and (value.func.value.id == 'torch') and (value.func.attr == 'sym_sum') and (len(value.args) == 1) and (not value.keywords) and isinstance(value.args[0], (ast.Tuple, ast.List)) and value.args[0].elts and all((integer(v) for v in value.args[0].elts)):
                parts = value.args[0].elts
                value = parts[0]
                for part in parts[1:]:
                    value = ast.BinOp(left=value, op=ast.Add(), right=part)
            value = Replace().visit(value)
            call = ast.unparse(value.func) if isinstance(value, ast.Call) else None

            def shape(node):
                if isinstance(node, ast.Name):
                    return shapes.get(node.id)
                if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) and (node.value.id == 'self') and (node.attr in scalar_literals):
                    return ()
                return None
            dimensions = None
            if call == 'torch.ops.aten.scalar_tensor.default':
                dimensions = ()
            if call == 'torch.ops.aten.lift_fresh_copy.default':
                dimensions = shape(value.args[0])
            if call == 'torch.ops.aten.expand.default' and isinstance(value.args[1], ast.List):
                dimensions = tuple((v.value if isinstance(v, ast.Constant) and type(v.value) is int and (v.value >= 0) else None for v in value.args[1].elts))
            if call == 'torch.ops.aten.where.self' and shape(value.args[1]) == () and (shape(value.args[2]) == ()):
                dimensions = shape(value.args[0])
            if call == 'torch.ops.aten.alias.default' and len(value.args) == 1 and (not value.keywords):
                replacements[name] = value.args[0]
                continue
            if call == 'torch.ops.aten.slice.Tensor' and len(value.args) in (4, 5) and (not value.keywords):
                dims = shape(value.args[0])
                if dims is not None and all((isinstance(v, ast.Constant) and type(v.value) is int for v in value.args[1:])) and (0 <= value.args[1].value < len(dims)) and (value.args[2].value == 0) and (dims[value.args[1].value] is not None) and (value.args[3].value >= dims[value.args[1].value]) and (len(value.args) == 4 or value.args[4].value == 1):
                    replacements[name] = value.args[0]
                    continue
            if call == 'torch.ops.aten.slice.Tensor' and (not value.keywords) and (len(value.args) in (4, 5)) and isinstance(value.args[1], ast.Constant) and (type(value.args[1].value) is int) and (0 <= value.args[1].value <= 4) and isinstance(value.args[2], ast.Constant) and (value.args[2].value == 0) and isinstance(value.args[3], ast.Constant) and (value.args[3].value == 9223372036854775807) and (len(value.args) == 4 or (isinstance(value.args[4], ast.Constant) and value.args[4].value == 1)):
                replacements[name] = value.args[0]
                continue
            if isinstance(value, ast.Attribute) and isinstance(value.value, ast.Name) and (value.value.id == 'self') or integer(value):
                replacements[name] = value
                continue
            shapes[f'v{len(rows)}'] = dimensions
            replacements[name] = ast.Name(id=f'v{len(rows)}', ctx=ast.Load())
            rows.append(ast.dump(value, include_attributes=False))
        elif isinstance(statement, ast.Return):
            rows.append(ast.dump(Replace().visit(statement), include_attributes=False))
        else:
            raise ValueError(f'Unexpected FX statement: {type(statement).__name__}')
    return rows
