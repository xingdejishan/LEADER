# Rotation coupling: two candidate fixes, hard subset, CPU only (20261022)

Subset: 1578 frames with front-end position error > 0.5 m + 2000 random control frames (seed 20261022) = 3578 frames.
Baseline = deployed branch (translation search with rotation frozen, then one rot block), reproducing results/full-online.npz bit-exactly (max |dpose| = 0); A = search->rot->search->rot; B = coarse stage per yaw in {-3..3} deg about world z (x forward, gravity z), best cell, fine stage, rot.

| band    | variant  |    n |  MPE |  MOE | ok(1m,2deg) | >1m | >5m |
|---------|----------|------|------|------|-------------|-----|-----|
| >0.5 m  | baseline | 1578 | 2.74 | 2.46 |      47.0 % | 342 | 277 |
| >0.5 m  | A        | 1578 | 2.68 | 2.42 |      47.5 % | 326 | 274 |
| >0.5 m  | B        | 1578 | 2.67 | 2.33 |      49.7 % | 315 | 273 |
| control | baseline | 2000 | 0.16 | 1.47 |      78.1 % |   0 |   0 |
| control | A        | 2000 | 0.15 | 1.52 |      76.6 % |   0 |   0 |
| control | B        | 2000 | 0.15 | 1.57 |      75.3 % |   0 |   0 |
| 5-8 m   | baseline |   43 | 4.50 | 2.20 |      23.3 % |  30 |  26 |
| 5-8 m   | A        |   43 | 4.14 | 2.36 |      23.3 % |  27 |  25 |
| 5-8 m   | B        |   43 | 3.96 | 2.28 |      23.3 % |  26 |  24 |
| all     | baseline | 3578 | 1.30 | 1.91 |      64.4 % | 342 | 277 |
| all     | A        | 3578 | 1.27 | 1.92 |      63.8 % | 326 | 274 |
| all     | B        | 3578 | 1.26 | 1.90 |      64.0 % | 315 | 273 |

**Verdict: no - neither variant fixes more of the 5-8 m band.** ok(1m,2deg) stays 10/43 for all three; A and B each repair 2 frames and break 2, and only the tail moves (>5 m 26 -> 25/24, MPE 4.50 -> 4.14/3.96).
Mechanism: 23 of the 43 never move (baseline included) - the truncated cost is 0.34 m at the front-end pose but 0.60 m at the true pose (0.59 m for the true translation under the front-end rotation), so the local reference cloud fits better at the wrong pose and no yaw offset or extra round of the same objective leaves that basin.
The other 20 the baseline already brings to 1.98 m mean (50 % ok), and B/A only shave that to 0.84/1.22 m - magnitude, not the ok count. B does lift the 0.5-2 m band (ok 57.3 -> 60.5 %, MPE 0.41 -> 0.36) but loses 56 control frames (78.1 -> 75.3 %, MOE 1.47 -> 1.57), since 46 % of control frames pick a non-zero yaw that lowers cost yet turns a good pose away; A is the same, smaller (+8 hard, -30 control).
**Cost.** Single process, per frame: baseline 0.164 s, A 0.304 s (+85 %), B 0.379 s (+131 %); 16-worker wall 246 s for 3578 frames (14.5 f/s); scan decode + map crop (~0.1 s, same for all) excluded.
**Artifacts.** /home/zhangcy/.dsh_scratch/rot_coupling_eval.py and /home/zhangcy/.dsh_scratch/rot-coupling-subset.npz(.json).
