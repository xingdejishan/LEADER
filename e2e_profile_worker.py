import sys
import time

import e2e_online as E


original_init = E.ParallelLocalizer.__init__
original_prepare = E.ParallelLocalizer.prepare_many
original_predict = E.ParallelLocalizer.predict_many


def initialize(self, *args, **kwargs):
    started = time.monotonic()
    original_init(self, *args, **kwargs)
    self.profile_initialization_seconds = time.monotonic() - started
    self.profile_prepared = {}


def prepare(self, specs):
    result = original_prepare(self, specs)
    finished = time.monotonic()
    for i, row in enumerate(result):
        end = result[i+1][4] if i+1 < len(result) else finished
        self.profile_prepared[row[0]['key']] = end
    return result


def predict(self, prepared):
    started = time.monotonic()
    rows = original_predict(self, prepared)
    self.frontend.torch.cuda.synchronize()
    finished = time.monotonic()
    for row, source in zip(rows, prepared):
        ready = self.profile_prepared.pop(row['key'])
        row['e2e_profile'] = dict(
            initialization_seconds=self.profile_initialization_seconds,
            prepare_started_at=source[4], prepared_at=ready,
            compute_started_at=started, final_T_ready_at=finished,
            preparation_service_seconds=ready-source[4],
            prefetch_queue_seconds=max(0., started-ready),
            batch_compute_seconds=finished-started,
            input_read_to_batch_T_seconds=finished-source[4],
            single_frame_service_seconds=(ready-source[4]+finished-started) if len(rows) == 1 else None,
            peak_torch_allocated_bytes=self.frontend.torch.cuda.max_memory_allocated(),
            peak_torch_reserved_bytes=self.frontend.torch.cuda.max_memory_reserved())
    return rows


E.ParallelLocalizer.__init__ = initialize
E.ParallelLocalizer.prepare_many = prepare
E.ParallelLocalizer.predict_many = predict


if __name__ == '__main__':
    sys.exit(E.main())
