"""Typed exact-object reuse for dead repeats; no native acceptance implied."""
import re


CONTRACT = 'original_Qwen_query1_proved_closed_unused_V_repeat_after_CSE_v1'


def validate_dead_repeat(sdk, diagnostics):
    marker = (r'^QWEN_GQA_DEAD_REPEAT erased=(\d+) shape-views=(\d+) '
              r'tensor-only-copy-proved=1 remaining-use-closure-proved=1 FP-arithmetic-changed=0$')
    if (sdk.get('dead_repeat_cleanup_requires_proved_closed_shape_use_DAG') is not True
            or sdk.get('original_K_compact_and_V_forwarding_retained') is not True
            or re.findall(marker,diagnostics,re.M) != [('28','84')]
            or diagnostics.index('QWEN_GQA_DEAD_REPEAT ') < diagnostics.rindex('QWEN_GQA_KEY_COMPACT ')):
        raise ValueError('Actual late dead28/views84 and original K proof required')
    for name in ('source','gqa_value_view_source','gqa_key_compact_source',
                 'gqa_value_forward_source','gqa_repeat_slice_source'):
        rows=[row for row in sdk['commands'] if '-c' in row['argv']
              and row['argv'][row['argv'].index('-c')+1]==sdk[name]['path']]
        enabled=name in ('gqa_value_forward_source','gqa_repeat_slice_source')
        if (len(rows)!=1 or rows[0]['returncode']!=0
                or rows[0]['argv'].count('-DQWEN_GQA_DEAD_REPEAT=1') != int(enabled)):
            raise ValueError('Only the two actual late-cleanup translation units may enable dead-repeat deletion')
