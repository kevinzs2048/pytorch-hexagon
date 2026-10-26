"""Observe native streamer events; never supply or choose generation tokens."""
import time


class NativeGenerationTimer:
    def __init__(self, *, clock=time.perf_counter_ns):
        self.clock = clock
        self.start = self.last = clock()
        self.prompt = None
        self.tokens, self.ready = [], []
        self.finished = None
        self.returned = None

    def _now(self):
        now = self.clock()
        if type(now) is not int or now < self.last:
            raise ValueError('Monotonic generation timing required')
        self.last = now
        return now

    def put(self, value):
        if self.finished is not None:
            raise ValueError('Token event after native generation ended')
        rows = value.tolist()
        now = self._now()
        if self.prompt is None:
            if not isinstance(rows, list) or len(rows) != 1 or not isinstance(rows[0], list) or not rows[0]:
                raise ValueError('Batch-one native prompt event required')
            self.prompt = list(rows[0])
        else:
            if not isinstance(rows, list) or len(rows) != 1 or type(rows[0]) is not int:
                raise ValueError('One actual native selected token required')
            self.tokens.append(rows[0])
            self.ready.append(now-self.start)

    def end(self):
        if self.finished is not None or not self.tokens:
            raise ValueError('Exactly one native end event after generated tokens required')
        self.finished = self._now()

    def completed(self, prompt, generated):
        if self.prompt != prompt or self.tokens != generated or self.finished is None or self.returned is None:
            raise ValueError('Observed native tokens differ from actual generation result')
        return dict(schema='native_generate_streamer_timing_v1', clock='perf_counter_ns',
            includes_capture_compiler_handoff_and_AOT_writeback=True,
            full_generate_ns=self.returned-self.start,
            native_stream_end_ns=self.finished-self.start,
            first_generated_token_ready_ns=self.ready[0],
            generated_token_ready_ns=list(self.ready),
            inter_token_ns=[b-a for a,b in zip(self.ready, self.ready[1:])],
            generated_tokens=len(generated), token_ids_verified=True)

    def mark_return(self):
        if self.finished is None or self.returned is not None:
            raise ValueError('One generate return after native streamer end required')
        self.returned = self._now()
