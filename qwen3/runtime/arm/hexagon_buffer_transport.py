"""Result checks for the opt-in buffer RPC, in addition to transport status."""
from hexagon_rpc_stream import HexagonRPCStream, PersistentRPCError


def buffer_capacities_for_contract(contract):
    """Size a session from an admitted positive, bounded tensor contract.

    The ABI admits only fixed positive dimensions and direct bounded variables,
    so the upper-bound corner maximizes every tensor and aligned buffer. Include
    the shape header in the transport cap, not only the tensor payload.
    """
    from qcom_hexagon_backend.backend.hexagon_dynamic_buffer_io import MAGIC, layout_for
    variables = contract['variables']
    if not 1 <= len(variables) <= 8 or any(
            type(bounds) not in (list, tuple) or len(bounds) != 2
            or any(type(value) is not int for value in bounds)
            or not 1 <= bounds[0] <= bounds[1] <= 1024
            for bounds in variables.values()):
        raise ValueError('Explicit bounded contract dimensions required')
    upper = {name: bounds[1] for name, bounds in variables.items()}
    layout = layout_for(contract, upper)
    incoming = len(MAGIC) + 8 * len(variables) + layout['input_bytes']
    outgoing = layout['output_bytes']
    if any(not 0 < size <= 256 * 1024**2 for size in (incoming, outgoing)):
        raise ValueError('Complete packet including header exceeds transport capacity')
    return dict(input_bytes=incoming, output_bytes=outgoing)


class HexagonBufferStream(HexagonRPCStream):
    def __init__(self, argv, env, directory, *, input_bytes, output_bytes, timeout=120, pass_fds=()):
        if any(type(n) is not int or not 0 < n <= 256*1024**2 for n in (input_bytes, output_bytes)):
            raise ValueError('Require bounded explicit buffer lengths')
        self.input_bytes = input_bytes; self.output_bytes = output_bytes
        super().__init__(argv, env, directory, timeout=timeout, pass_fds=pass_fds)

    def start(self):
        event = super().start()
        if (type(event.get('input_bytes')) is not int or type(event.get('output_bytes')) is not int
                or event['input_bytes'] != self.input_bytes or event['output_bytes'] != self.output_bytes):
            raise PersistentRPCError('ARM buffer lengths differ from compiled ABI')
        return event

    def invoke(self):
        return self.validate_invoke(super().invoke())

    def validate_invoke(self, event):
        if (type(event.get('DSP_result')) is not int or event['DSP_result'] != 0
                or type(event.get('output_result')) is not int or event['output_result'] != 0
                or type(event.get('kernel_us')) is not int or event['kernel_us'] < 0):
            raise PersistentRPCError('Fresh buffer output write/timing invalid: '+str(event))
        if ('AP_input_fnv1a32' in event or 'DSP_input_fnv1a32' in event) and (
                type(event.get('AP_input_fnv1a32')) is not int or type(event.get('DSP_input_fnv1a32')) is not int
                or not 0 <= event['AP_input_fnv1a32'] <= 0xffffffff
                or event['AP_input_fnv1a32'] != event['DSP_input_fnv1a32']):
            raise PersistentRPCError('ARM/DSP observed input bytes differ: '+str(event))
        if 'RPC_input_fnv1a32' in event and event['RPC_input_fnv1a32']!=event.get('AP_input_fnv1a32'):
            raise PersistentRPCError('ARM/DSP RPC-entry observed input bytes differ: '+str(event))
        return event


class HexagonDynamicBufferStream(HexagonBufferStream):
    """One loaded DSO with per-call byte lengths bounded by allocated caps."""

    def invoke(self):
        raise ValueError('Bounded dynamic buffer calls require explicit byte lengths')

    def start(self):
        event = super().start()
        if event.get('dynamic_sizes') is not True:
            raise PersistentRPCError('ARM buffer runtime did not opt into dynamic sizes')
        return event

    def invoke_sized(self, input_bytes, output_bytes):
        if (type(input_bytes) is not int or type(output_bytes) is not int
                or not 0 < input_bytes <= self.input_bytes
                or not 0 < output_bytes <= self.output_bytes):
            raise ValueError('Dynamic invocation exceeds session buffer capacities')
        command = f'invoke {input_bytes} {output_bytes}\n'.encode('ascii')
        event = self.validate_invoke(self._invoke(command))
        if (event.get('input_bytes') != input_bytes
                or event.get('output_bytes') != output_bytes):
            raise PersistentRPCError('ARM dynamic invocation sizes differ from request')
        return event


class HexagonLibraryBufferStream(HexagonDynamicBufferStream):
    """Opt-in loaded-library slots; tensors and generation remain caller-owned.

    The caller must pin/stage immutable libraries before selecting their paths.
    This protocol does not assert that the loader actually shares weights.
    """

    @staticmethod
    def validate_library_path(path):
        if (type(path) is not str or not path.startswith('/dev/shm/qwen3-')
                or len(path)>500 or '..' in path
                or any(ord(c)<=32 or ord(c)>=127 for c in path)):
            raise ValueError('Explicit bounded staged library path required')

    def __init__(self, *args, initial_library, **kwargs):
        self.validate_library_path(initial_library)
        self.libraries=[initial_library]
        self.current_library=0
        self.library_owners=[dict(index=0,loaded=True)]
        super().__init__(*args, **kwargs)

    def start(self):
        event=super().start()
        if event.get('library_switching') is not True:
            raise PersistentRPCError('Runtime did not opt into library switching')
        return event

    def select_library(self, path):
        self.validate_library_path(path)
        if not self.ready or self.closed:
            raise PersistentRPCError('Stream not ready')
        cached=path in self.libraries
        index=self.libraries.index(path) if cached else len(self.libraries)
        if index>=4:
            raise ValueError('Bounded library slots exhausted; no silent eviction')
        self.process.stdin.write(('select '+path+'\n').encode('ascii'))
        self.process.stdin.flush()
        if not cached:
            event=self.event('entry_open')
            if type(event.get('library_slot')) is not int or event['library_slot']!=index:
                raise PersistentRPCError('Library load ownership differs')
            owner=dict(index=index,loaded=False)
            self.library_owners.append(owner)
            try:
                event=self.event('entry_load')
            finally:
                # A loader failure still owns a worker when the RPC succeeded.
                # Conversely an RPC failure owns only the opened context.
                last=self.events[-1]
                owner['loaded']=(last.get('stage')=='entry_load'
                    and type(last.get('rpc_result')) is int and last['rpc_result']==0
                    and type(last.get('library_slot')) is int and last['library_slot']==index)
            if type(event.get('library_slot')) is not int or event['library_slot']!=index:
                raise PersistentRPCError('Library load ownership differs')
            self.libraries.append(path)
        event=self.event('select')
        if (type(event.get('library_slot')) is not int or event['library_slot']!=index
                or event.get('cached') is not cached):
            raise PersistentRPCError('Library selection identity differs')
        self.current_library=index
        return event

    def invoke_sized(self, input_bytes, output_bytes):
        event=super().invoke_sized(input_bytes, output_bytes)
        if (type(event.get('library_slot')) is not int
                or event['library_slot']!=self.current_library):
            raise PersistentRPCError('Invocation executed a different library slot')
        return event

    def _consume_close_events(self):
        for owner in reversed(self.library_owners):
            index=owner['index']
            for stage in (('unload','close') if owner['loaded'] else ('close',)):
                event=self.event(stage)
                if type(event.get('library_slot')) is not int or event['library_slot']!=index:
                    raise PersistentRPCError('Library cleanup ownership differs')
