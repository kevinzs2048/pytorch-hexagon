from models.qwen3.runtime.arm.hexagon_dynamic_cache_graph import canonical


def test_request_order_can_split_equal_length_symbols():
    a = '''def forward(self, n, ids, positions):
 x=torch.ops.aten.view.default(ids,[1,n])
 y=torch.ops.aten.view.default(positions,[1,n])
 return (x,y)
'''
    b = '''def forward(self, positions, m, unused_stride, ids, n):
 x=torch.ops.aten.view.default(ids,[1,n])
 y=torch.ops.aten.view.default(positions,[1,m])
 return (x,y)
'''
    sa = ["L['input_ids'].size()[1]", "L['input_ids']", "L['cache_position']"]
    sb = [sa[2], "L['cache_position'].size()[0]",
          "L['attention_mask'].stride()[0]", sa[1], sa[0]]
    assert canonical(a, sa) == canonical(b, sb)
    # Equal values alone do not justify swapping token and position tensors.
    swapped = [sa[1], *sb[1:3], sa[2], sa[0]]
    assert canonical(a, sa) != canonical(b, swapped)


def test_integer_aliases_only():
    a='def forward(self, n, x):\n s=n+1\n y=torch.ops.aten.view.default(x,[s])\n return y\n'
    b='def forward(self, n, m, x):\n s=torch.sym_sum((m,1))\n y=torch.ops.aten.view.default(x,[s])\n return y\n'
    sa=["L['past_key_values'].layers[0].keys.size()[2]", "L['input_ids']"]
    sb=[sa[0],"L['past_key_values'].layers[1].values.size()[2]",sa[1]]
    assert canonical(a,sa)==canonical(b,sb)
    assert canonical(a,sa)!=canonical(b.replace('(m,1)','(m,2)'),sb)
    assert canonical(a,sa)!=canonical(b.replace('aten.view.default','aten.neg.default'),sb)


def test_only_proven_whole_mask_slice_matches_alias():
    prefix='''def forward(self, x):
 a=torch.ops.aten.expand.default(x,[1,1,1,2])
 b=self.literal
 c=torch.ops.aten.lift_fresh_copy.default(b)
 d=torch.ops.aten.scalar_tensor.default(-1.0)
 e=torch.ops.aten.where.self(a,c,d)
'''
    slice_code=prefix+' y=torch.ops.aten.slice.Tensor(e,3,0,2)\n return y\n'
    alias_code=prefix+' y=torch.ops.aten.alias.default(e)\n return y\n'
    src=["L['attention_mask']"]
    assert canonical(slice_code,src,scalar_literals=['literal'])==canonical(alias_code,src,scalar_literals=['literal'])
    assert canonical(slice_code,src)!=canonical(alias_code,src)
    assert canonical(slice_code.replace('(e,3,0,2)','(e,3,0,1)'),src,scalar_literals=['literal'])!=canonical(alias_code,src,scalar_literals=['literal'])
    assert canonical(slice_code.replace('(e,3,0,2)','(e,3,1,2)'),src,scalar_literals=['literal'])!=canonical(alias_code,src,scalar_literals=['literal'])


def test_floating_point_arithmetic_is_kept_in_order():
    a='def forward(self,x):\n y=x+1.0\n z=y+2.0\n return z\n'
    b='def forward(self,x):\n y=x+2.0\n z=y+1.0\n return z\n'
    assert canonical(a,["L['input_ids']"])!=canonical(b,["L['input_ids']"])


def test_invalid_graph_contract_is_rejected_under_python_optimization():
    import subprocess
    import sys
    cases = [('def forward(self, x):\n return x\n', []),
             ('def forward(self, x):\n a=b=x\n return a\n', ["L['input_ids']"])]
    code = ("from models.qwen3.runtime.arm.hexagon_dynamic_cache_graph import canonical\n"
            + f"cases = {cases!r}\n" + """
for source, sources in cases:
    try:
        canonical(source, sources)
    except ValueError:
        continue
    raise SystemExit('Invalid graph was admitted')
""")
    subprocess.run([sys.executable, '-O', '-c', code], check=True)
