# Light selection trial (2026-10-24)

Experimental opt-in only. Default remains full (e2e_online.py:506).
96 identical live-generated candidate pools, including 24 historically visual-adopted stress samples; not a random/full-test accuracy estimate.

Selection latency: full replay mean 1041.88 ms, P95 1966.08 ms. Light active decisions (36/96): mean 66.26 ms, P95 95.57 ms, max 113.06 ms. Other 60 frames have no qualified visual proposals and fall back. Not end-to-end latency; mean live stage work 1.034 s/frame.

Fixed-output offline truth evaluation: full/light translation P95 2.026/9.307 m; over-5m 3/8 frames; over-10m 1/4 frames. Nine choice changes. Key 2012-05-26/1338074839327574 worsens from 1.743 m to 442.571 m. Truth read only after predictions and parameters fixed; no subsequent tuning.

Do not promote: qualification/Top2 prefilter discards necessary rescue candidates. Next proposed experiment: retain candidate diversity and batch bounded sparse geometric scoring, evaluated on a separate validation split.

Artifacts: results/light_selection_comparison_20261024.json and results/light_selection_accuracy_20261024.json.
