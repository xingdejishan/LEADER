import copy
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from raybundle.async_runtime import restore_training,publish_job,verified_job,pending_jobs
from raybundle.mapping import sha256


class AsyncRuntimeTests(unittest.TestCase):
    def test_resume_preserves_next_update_and_random_order(self):
        torch.manual_seed(20)
        model=torch.nn.Linear(3,2).double()
        optimizer=torch.optim.Adam(model.parameters(),lr=1e-4)
        rng=np.random.RandomState(20)
        def step():
            optimizer.zero_grad()
            model(torch.randn(4,3,dtype=torch.float64)).square().sum().backward()
            optimizer.step()
        step()
        rng.permutation(1200)
        saved=copy.deepcopy(dict(model=model.state_dict(),optimizer=optimizer.state_dict(),
            numpy_rng=rng.get_state(),torch_rng=torch.get_rng_state(),cuda_rng=[]))
        expected_order=rng.permutation(1200)
        step()
        expected=copy.deepcopy(model.state_dict())
        restore_training(saved,model,optimizer,rng)
        np.testing.assert_array_equal(rng.permutation(1200),expected_order)
        step()
        for key,value in model.state_dict().items():
            self.assertTrue(torch.equal(value,expected[key]))

    def test_queue_rejects_changed_and_duplicate_checkpoints(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            (root/'queue').mkdir()
            (root/'results').mkdir()
            checkpoint=root/'02.pt'
            checkpoint.write_bytes(b'frozen')
            digest=sha256(checkpoint)
            publish_job(root,2,checkpoint,digest,dict(updates=300))
            jobs=pending_jobs(root)
            self.assertEqual(len(jobs),1)
            self.assertEqual(verified_job(jobs[0])['checkpoint_sha256'],digest)
            with self.assertRaises(FileExistsError):
                publish_job(root,2,checkpoint,digest,{})
            checkpoint.write_bytes(b'changed')
            with self.assertRaises(ValueError):
                verified_job(jobs[0])
            (root/'results'/'02.json').write_text('{}')
            self.assertEqual(pending_jobs(root),[])


if __name__=='__main__':
    unittest.main()
