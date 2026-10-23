# MoE relocaliser — complete version

A LiDAR relocalisation chain built around one number. Everything below is measured on
the NCLT test split; nothing here is trained by this repository, and nothing is cached
across frames.

```
C(T) = mean over the scan of min( dist(T x, M), 1.0 m )        M = local reference cloud

T_leader (front end)
   │
   ├─ coarse surface S(s, d), 17 x 3, built once per frame ……………… stage A
   │     ├─ descent depth  g = C(T0) − min S          → is the search worth running?
   │     ├─ ambiguity ratio ρ = (C_far − C0)/C0       → is another modality worth asking?
   │     └─ adaptive fine stage → T3 (translation only; rotation preserved)
   │
   ├─ rotation block on T3 (translation frozen, accepted on a held-out half)
   │
   └─ confidence gate                                 → keep or drop the translation
          ↓
     visual hypothesis  ──→  arbitration: C_visual < C_route3 − 0.05 → switch
```

Five decisions, one cost, and each decision is a statement about the same statistic.

## Results

Full test split, 104,728 frames. Leave one test date out for the gate; the leader and the
search are deterministic.

| policy | MPE | MOE | ok(1 m, 2°) | >1 m | >5 m | frames made worse |
|---|---|---|---|---|---|---|
| leader front end, no correction | 0.18755 | 1.404661 | 80.20% | 685 | 290 | — |
| accept every correction | 0.18433 | 1.401758 | 80.59% | 366 | 284 | 4051 |
| **with the confidence gate** | **0.18041** | **1.401758** | **80.60%** | **353** | **284** | **1501** |

On the 1502 hard frames, where the visual table is available:

| policy | MPE | MOE | ok | >5 m | >10 m |
|---|---|---|---|---|---|
| front end (refined V7) | 1.9130 | 2.5234 | 621 | — | 8 |
| geometry only | 1.5056 | 2.4839 | 746 | 19 | 7 |
| geometry + visual arbitration | 0.8640 | 2.1663 | 754 | 9 | 3 |
| vision alone, always its best candidate | 23.3105 | 12.9626 | 342 | 372 | 306 |
| oracle, whichever is closer | 0.7904 | 2.1937 | 793 | 4 | 3 |

Vision is the better hypothesis on 25% of the hard frames and unusable on its own; the
arbitration resolves 97% of the frames where the two disagree by more than a metre.

## The confidence gate

The search minimises a scan-to-map cost, so a correction can lower that cost while
moving away from the truth — the scan and the map share the same bias. The gate
re-measures the same gain on disjoint evidence: eight azimuth sectors of the scan, each
scored separately, and each of the three reference clouds on its own. A real correction
reproduces on most sectors and on every reference; one driven by map bias does not.

Ranking quality on the 7,719 frames where the search runs: AUC 0.766 leave-one-date-out,
against 0.545 for every feature the pipeline already computed; sector and reference
consistency alone reach 0.725. Shuffled-label controls return 0.497 / 0.518 / 0.468.
Calibration on held-out scores is monotone: the lowest decile is 7% damaged and improves
by 0.79 m on average, the highest is 85% damaged and loses 0.14 m.

The gate acts on the translation only. The rotation block runs on every searched frame,
which is what keeps the mean orientation error bit-identical.

Cost: 27 ms per searched frame for the sector and reference features (104 ms more if the
normal-based features are enabled). Searched frames are 7.4% of the split.

## Layout

    algorithms/            CFS (search) and MoE (whole chain), numpy + scipy only
    confidence.py          the cross-view consistency features, gate and coefficients
    confidence-coefficients.json
    three_route_unified.py offline orchestrator over the 1502 hard frames
    three_route.py         the same chain, single file, no arbitration
    run_pipeline.py        one entry point: frame table + initial poses -> poses
    e2e_online.py          online driver (needs the LEADER runtime and a GPU)
    visual_branch.py       live visual branch: retrieval + matching + PnP
    along_track_dp.py      sequence DP experiment (negative result, kept for the record)
    stages/                coarse/fine search, ambiguity ratio, gate1
    inputs/                the 1502 hard frames, the 4752-reference library, initial poses
    docs/                  design notes; docs/notes/ the measurement reports

## Running

Offline, CPU only, needs `inputs/` and the NCLT scans at the paths inside them:

```bash
python three_route_unified.py --workers 8            # arbitration path
python three_route_unified.py --mode route3          # geometry only
python run_pipeline.py --frames 1502 --workers 8     # one entry point
```

Online (`e2e_online.py`) additionally needs the frozen LEADER runtime, the appearance
field checkpoint and a GPU; it is the code that produced the full-split numbers and it
writes a per-frame ledger. `--confidence-gate 1` enables the gate.

## Status and caveats

- The gate was verified against the offline implementation on the geometric stage
  (scores agree to 2·10⁻⁹, identical decisions) and on a 40-frame pipeline run with no
  skips. A full-split certification run with the gate enabled is **not** in this tree
  yet: it needs 22 hours of GPU and the machine was busy. The 0.18041 row comes from
  replaying the same geometric code over the stored search results.
- `tests/test_no_rotation.py` is stale: it asserts the entry points contain no rotation
  module, which was true for one day in October and is not the current design. It is
  excluded rather than fixed.
- `e2e_online.py` imports the private LEADER runtime; it is included because it is the
  driver that produced the numbers, not because it runs standalone.
- The 1502-frame results use a replaced-reference truth for one window of 2012-02-12
  (255 frames whose official ground truth is locally deformed). The full-split tables
  report both conventions where it matters; see `docs/notes/gate-and-truth-conventions-20261022.md`.

## Where the numbers come from

`docs/notes/` holds the measurements, each with its script and its failures:

- `discriminator-20261023.md` — the confidence gate, its ablation, controls, calibration
- `vision-in-search-20261023.md` — why vision cannot be a search prior, and what it is worth
- `shared-map-arbitration-20261022.md` — sharing one map between hypotheses costs 0.6 m
- `rotation-coupling-two-fixes-20261022.md` — two attempted fixes for the rotation coupling
- `truncation-cost-test-20261022.md` — nine cost-function variants, all neutral or worse
- `along-track-dp.md` — sequence DP over the along-track offsets
- `sweep.md` — the 67-row threshold sweep
- `moe-diagnosis-20261022.md`, `gate-and-truth-conventions-20261022.md`
