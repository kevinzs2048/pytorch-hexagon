"""Retain an existing memcpy wrapper by symbol-only rename and exact text verification."""
import hashlib
from pathlib import Path
import subprocess
import tempfile


def sha(path):
    with Path(path).open('rb')as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def preserve_wrapped_memcpy(command, nm, objcopy, output,
                            delegate_symbol='qwen_profile_preserved_memcpy'):
    """Let diagnostic counters delegate to an existing compiled memcpy wrapper.

    Only its exported symbol is renamed. The original object stays untouched,
    and the complete .text section must be identical before linking.
    """
    if delegate_symbol not in ('qwen_profile_preserved_memcpy','qwen_global_copy_preserved_memcpy'):
        raise ValueError('An explicit reviewed memcpy delegate symbol is required')
    if '-Wl,--wrap=memcpy' not in command:
        return None
    candidates=[]
    for word in command:
        path=Path(word)
        if path.is_absolute() and path.suffix=='.o' and path.is_file():
            symbols=subprocess.check_output([str(nm),'--defined-only',str(path)],text=True)
            names={line.split()[-1] for line in symbols.splitlines() if line.split()}
            if delegate_symbol in names:
                raise ValueError('Diagnostic delegate symbol already exists')
            if '__wrap_memcpy' in names:
                candidates.append(path)
    if len(candidates)!=1:
        raise ValueError('Exactly one compiled original memcpy wrapper required')
    original=candidates[0];original_sha=sha(original)
    if command.count(str(original))!=1:
        raise ValueError('Original memcpy object must occur exactly once')
    renamed=output/'preserved-memcpy.o'
    if renamed.exists():
        raise ValueError('Fresh diagnostic delegate object required')
    rename_command=[str(objcopy),'--redefine-sym',
        '__wrap_memcpy='+delegate_symbol,str(original),str(renamed)]
    subprocess.run(rename_command,check=True,capture_output=True)
    names={line.split()[-1] for line in subprocess.check_output(
        [str(nm),'--defined-only',str(renamed)],text=True).splitlines() if line.split()}
    if '__wrap_memcpy' in names or delegate_symbol not in names:
        raise ValueError('Original wrapper export was not renamed exactly')
    with tempfile.TemporaryDirectory(prefix='qwen-profile-text-') as temporary:
        folder=Path(temporary);texts=[]
        for index,path in enumerate((original,renamed)):
            text=folder/(str(index)+'.text')
            subprocess.run([str(objcopy),'--dump-section','.text='+str(text),
                str(path),str(folder/(str(index)+'.o'))],check=True,capture_output=True)
            texts.append(text.read_bytes())
        if not texts[0] or texts[0]!=texts[1]:
            raise ValueError('Original memcpy machine instructions changed')
        import hashlib
        text_sha=hashlib.sha256(texts[0]).hexdigest();text_bytes=len(texts[0])
    if sha(original)!=original_sha:
        raise ValueError('Frozen original memcpy object changed')
    command[command.index(str(original))]=str(renamed)
    return dict(original_object=dict(path=str(original),sha256=original_sha),
        diagnostic_delegate_object=dict(path=str(renamed),sha256=sha(renamed)),
        original_text_bytes=text_bytes,original_text_sha256=text_sha,
        original_text_instructions_identical=True,original_object_unchanged=True,
        delegate_symbol=delegate_symbol,rename_command=rename_command,
        tools=[dict(path=str(p),sha256=sha(p)) for p in (nm,objcopy)])

