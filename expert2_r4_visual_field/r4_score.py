"""Score the R4 outputs with the canonical unified scorer, hard1502 only.

The ordinary1000 group is passed as its own frozen baseline (we never touch ordinary
frames), so the printed hard rows are exactly what score_unified.py computes for our
prediction. Also writes a combined npz (hard predictions + untouched ordinary baseline)
so ``score_unified.py --pred`` can be run by hand.
"""
import sys

import numpy as np

OUT = '/data2/zhangcy/r4-full-20261019-v1'
sys.path.insert(0, '/home/zhangcy/projects/r-unified-eval-20261018-v1')
import score_unified as SU  # noqa: E402

w = SU.load_world()
for tag in sys.argv[1:] or ['main', 'main_refined']:
    z = np.load('%s/poses_%s.npz' % (OUT, tag), allow_pickle=True)
    look = {str(k): i for i, k in enumerate(z['keys'])}
    ph = np.asarray(z['poses'], float)[[look[k] for k in w['hard_keys']]]
    rows = SU.evaluate(ph, w['ord_base'], w, 'R4 %s' % tag)
    full = np.concatenate([ph, w['ord_base']])
    np.savez_compressed('%s/poses_%s_with_ord.npz' % (OUT, tag),
                        keys=np.array(w['hard_keys'] + w['ord_keys']), poses=full)
    print('wrote %s/poses_%s_with_ord.npz' % (OUT, tag))
