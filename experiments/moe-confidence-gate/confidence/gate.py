"""Decision rule on top of the consistency features.

The score is a logistic regression over the feature vector, standardised with the
training mean and deviation. Lower is better: a low score means the correction looks
consistent across sectors and references, so it is accepted. Very large corrections are
accepted regardless of the score, because they are the reliable ones on average and
they hold the long tail down.

Reported result, leave one test date out: 3.9 mm lower mean position error than
accepting every correction, 63% fewer frames made worse, orientation unchanged.
"""
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT = os.path.join(HERE, 'coefficients.json')


class Gate:
    def __init__(self, path=DEFAULT):
        with open(path) as stream:
            blob = json.load(stream)
        self.features = list(blob['features'])
        self.mean = np.asarray(blob['mean'], float)
        self.std = np.asarray(blob['std'], float)
        self.weights = np.asarray(blob['weights'], float)
        self.bias = float(blob['bias'])
        policy = blob.get('policy', {})
        self.score_below = float(policy.get('score_below', 0.3))
        self.always_above = float(policy.get('always_accept_shift_above', 1.5))

    def vector(self, feats):
        x = np.asarray([feats.get(f, np.nan) for f in self.features], float)
        x = np.where(np.isfinite(x), x, self.mean)          # missing input -> training mean
        return (x - self.mean) / self.std

    def score(self, feats):
        return float(self.vector(feats) @ self.weights + self.bias)

    def accept(self, feats):
        """True when the geometric correction should be applied."""
        score = self.score(feats)
        shift = feats.get('shift', 0.0)
        shift = 0.0 if shift is None or not np.isfinite(shift) else float(shift)
        return bool(score < self.score_below or shift > self.always_above), score
