import hashlib


def patch_training_vendor(vendor):
    path = vendor / 'ace_trainer.py'
    source = path.read_text()
    original_hash = hashlib.sha256(path.read_bytes()).hexdigest()
    if 'self.optimizer._step_count' not in source:
        return dict(applied=False)
    marker = '        self.optimizer = optim.AdamW(self.regressor.heads.parameters(), lr=self.options.learning_rate_min)'
    if source.count(marker) != 1 or source.count('class TrainerACE:\n') != 1:
        raise ValueError('Unexpected trainer implementation')
    source = source.replace('class TrainerACE:\n',
        'class TrainerACE:\n'
        '    def _record_optimizer_step(self, optimizer, args, kwargs):\n'
        '        self.optimizer_steps += 1\n\n')
    source = source.replace(marker, marker + '\n'
        '        self.optimizer_steps = 0\n'
        '        self.optimizer.register_step_post_hook(self._record_optimizer_step)')
    source = source.replace('self.optimizer._step_count', 'self.optimizer_steps')
    path.write_text(source)
    return dict(applied=True, reason='Count actual AMP optimizer updates through the public optimizer hook',
        original_sha256=original_hash, patched_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
