import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

root = Path('/root/rivermind-data/glace_nclt_corrected_20260912')
source = Path('/root/rivermind-data/glace_nclt_full_20260912/vendor')
vendor = root / 'vendor_corrected'
vendor.mkdir(exist_ok=True)
for path in source.glob('*.py'):
    (vendor / path.name).write_text(path.read_text())
if not (vendor / 'datasets').exists():
    shutil.copytree(source / 'datasets', vendor / 'datasets', ignore=shutil.ignore_patterns('__pycache__'))
if not (vendor / 'ace_encoder_pretrained.pt').exists():
    (vendor / 'ace_encoder_pretrained.pt').symlink_to(source / 'ace_encoder_pretrained.pt')
path = vendor / 'ace_trainer.py'
text = path.read_text()
old = '        for k, v in head_state_dict.items():\n            head_state_dict[k] = head_state_dict[k].half()'
assert old in text
text = text.replace(old, '        head_state_dict = {k: v.detach().float().cpu() for k, v in head_state_dict.items()}')
text = text.replace('        torch.save(head_state_dict, self.options.output_map_file)',
                    '        temporary = str(self.options.output_map_file) + ".tmp"\n'
                    '        torch.save(head_state_dict, temporary)\n'
                    '        os.replace(temporary, self.options.output_map_file)')
text = text.replace('        if self.iteration % self.iterations_output == 0:',
                    '        if self.iteration > 0 and self.iteration % 5000 == 0 and dist.get_rank() == 0:\n'
                    '            self.save_model()\n'
                    '        if self.iteration % self.iterations_output == 0:')
path.write_text(text)
head = root / 'glace_head.pt'
train_args = ['--training_buffer_size', '5505536', '--samples_per_image', '128', '--batch_size', '8192',
              '--max_iterations', '60000', '--image_resolution', '616', '--aug_rotation', '0', '--aug_scale', '1',
              '--feat_noise_std', '0', '--num_decoder_clusters', '64', '--head_channels', '768',
              '--repro_loss_soft_clamp', '300']
config = {'train_args': train_args, 'train_images': 43012, 'test_images': 0,
          'extrinsic': 'correct camera-to-body product, matched to official NCLT script',
          'head_precision': 'float32', 'seed': 2089, 'vendor': str(vendor), 'started': time.time()}
(root / 'retrain_config.json').write_text(json.dumps(config, indent=2))
command = [sys.executable, '-m', 'torch.distributed.run', '--rdzv_backend', 'c10d', '--rdzv_endpoint',
           'localhost:29529', '--rdzv_id', 'corrected_full', '--nnodes', '1', '--nproc_per_node', '1',
           str(vendor / 'train_ace.py'), str(root / 'scene'), str(head)] + train_args
with (root / 'train.log').open('w') as log:
    code = subprocess.run(command, cwd=vendor, stdout=log, stderr=subprocess.STDOUT).returncode
state = {'returncode': code, 'finished': time.time(), 'head': str(head), 'ready_for_test': False}
if code == 0:
    state['head_sha256'] = hashlib.sha256(head.read_bytes()).hexdigest()
(root / 'training_finished.json').write_text(json.dumps(state, indent=2))
print(json.dumps(state), flush=True)
