import unittest

import numpy as np
from scipy.spatial.transform import Rotation

from .pairwise_camera_ranking import candidates, pair_counts


class PairwiseRankingTests(unittest.TestCase):
    def test_candidate_errors_and_decoupling(self):
        gt = np.eye(4)
        gt[:3,:3] = Rotation.from_rotvec([.2,.3,.4]).as_matrix()
        gt[:3,3] = [100.,-200.,10.]
        for mode in ['translation','rotation']:
            poses = candidates(gt,mode)
            self.assertEqual(len(poses),25)
            for item in poses:
                pose = item['pose']
                translation = np.linalg.norm(pose[:3,3]-gt[:3,3])
                rotation = np.rad2deg(Rotation.from_matrix(pose[:3,:3]@gt[:3,:3].T).magnitude())
                self.assertAlmostEqual(translation,item['level'] if mode=='translation' else 0.)
                self.assertAlmostEqual(rotation,item['level'] if mode=='rotation' else 0.)

    def test_unequal_magnitude_pairs_only(self):
        poses = candidates(np.eye(4),'translation')
        scored = [dict(item,S_C=item['level']) for item in poses]
        result = pair_counts(scored)
        self.assertEqual(result['pairs'],240)
        self.assertEqual(result['accuracy'],1.)
        self.assertEqual(pair_counts(scored,same_direction=True)['pairs'],60)
        self.assertEqual(pair_counts(scored,levels=(.2,.5))['pairs'],36)
        self.assertEqual(pair_counts(scored,levels=(.2,.5),same_direction=True)['pairs'],6)

    def test_ties_not_counted_correct(self):
        scored = [dict(item,S_C=1.) for item in candidates(np.eye(4),'rotation')]
        result = pair_counts(scored)
        self.assertEqual(result['ties'],240)
        self.assertEqual(result['accuracy'],0.)
        self.assertEqual(result['half_credit_accuracy'],.5)


if __name__ == '__main__':
    unittest.main()
