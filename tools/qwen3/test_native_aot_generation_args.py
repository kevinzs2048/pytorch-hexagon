"""Request changes preserve accepted runtime and compiler arguments."""
import json
import shlex
import pytest
from generate_native_aot import generation_command


def baseline():
    return ['env','PYTHONDONTWRITEBYTECODE=1','PYTHONPATH=/sources','/python','/native.py',
        '--catalog','/catalog','--catalog-sha256','catalog-sha','--context','/context','--hf','/hf',
        '--decompositions','/decomp','--prompts-json','["old","old2"]','--max-new-tokens','8',
        '--chunk-size','32','--output','/old-output','--workdir','/dev/shm/old','--compiler-mailbox','/old-mailbox',
        '--parameter-check','exact-bytes','--specialize-floats','--input-packer','joined',
        '--constant-load-split-proofs','/proofs','--constant-loader-mode','verified-payload-preserving',
        '--local-packet-transport','shared-memory','--output-decoder','owned-once','--packet-recording','hashes']


def options(**changes):
    x=dict(prompts='["new"]',requests=None,max_new_tokens=64,chunk_size=32,
        chat_template=False,output='/new-output',workdir='/dev/shm/new',mailbox='/new-mailbox');x.update(changes);return x


@pytest.mark.parametrize('kind',['text','token-ids','chat'])
def test_input_changes_preserve_all_runtime_guards(kind):
    original=baseline();before=list(original)
    opts=options(**({'prompts':None,'requests':'[[5]]'} if kind=='token-ids' else {'chat_template':kind=='chat'}))
    result=generation_command(original,**opts)
    assert original==before
    for flag in ('--catalog','--catalog-sha256','--context','--hf','--decompositions','--parameter-check',
        '--input-packer','--constant-load-split-proofs','--constant-loader-mode','--local-packet-transport',
        '--output-decoder','--packet-recording'):
        assert result[result.index(flag)+1]==original[original.index(flag)+1]
    assert '--specialize-floats' in result and result.count('--output')==1
    assert '--chat-template' in result if kind=='chat' else '--chat-template' not in result
    assert result[result.index('--max-new-tokens')+1]=='64'
    assert '--prompts-json' not in result if kind=='token-ids' else '--requests-json' not in result


def test_prompt_quotes_and_shell_syntax_roundtrip_as_literal_data():
    prompt=json.dumps(['中文代码 `cmd` $(cmd) \' " ;\\n\n新行'])
    result=generation_command(baseline(),**options(prompts=prompt))
    assert shlex.split(shlex.join(result))==result
    assert result[result.index('--prompts-json')+1]==prompt


@pytest.mark.parametrize('changes',[{'prompts':None},{'requests':'[[5]]'},{'prompts':None,'requests':'[[5]]','chat_template':True}])
def test_rejects_ambiguous_input_and_token_chat(changes):
    with pytest.raises(ValueError):generation_command(baseline(),**options(**changes))


def test_removes_existing_input_and_chat_options_for_new_request():
    original=baseline()+['--chat-template','--requests-json','[[6]]']
    result=generation_command(original,**options(prompts=None,requests='[[5]]'))
    assert result.count('--requests-json')==1 and '--prompts-json' not in result and '--chat-template' not in result


def test_rejects_incomplete_original_option():
    with pytest.raises(ValueError):generation_command(baseline()+['--output'],**options())


def test_original_warm_mode_has_no_mailbox_or_live_startup_option():
    original=baseline()+['--startup-timing','--generation-timing']
    result=generation_command(original,**options(mailbox=None))
    assert '--compiler-mailbox' not in result and '--startup-timing' not in result
    assert '--generation-timing' in result and '--specialize-floats' in result
    assert result[result.index('--parameter-check')+1]=='exact-bytes'
