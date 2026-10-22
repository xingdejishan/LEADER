# Rotation diagnostics: where the MOE actually comes from

Four small scripts that answer one question at four depths: **why is the rotation
error what it is, and what would it take to lower it?**

They are read-only. They load the frozen scorer (which owns the truth policy and the
reference substitution), the current model's inputs, and the search stages the model
actually runs. Nothing is written except `yaw_reach.json`.

```bash
python rot_decomp.py                      # axis decomposition of the deployed baseline
python rot_decomp.py results/model.npz    # same, for any prediction table
python rot_worth.py                       # counterfactual MOE if one axis were perfect
python yaw_ident.py                       # is the yaw identifiable at the TRUE position
python yaw_reach.py                       # is the correct yaw reachable from the model's OWN position
```

Edit `config.py` first; it holds every external path.

## 1. `rot_decomp.py` — the axis budget

Rotation error is `R_pred^T R_gt`, expressed in the body frame, then reported per
axis from its axis-angle vector. On the 1502 hard frames, deployed baseline
(mean angle 2.5235 deg):

| axis | mean abs | share of squared error | median abs |
|---|---|---|---|
| roll  | 1.05 deg | 25.6% | 0.66 deg |
| pitch | 1.12 deg | 6.9%  | 0.89 deg |
| yaw   | 1.47 deg | **67.5%** | 0.81 deg |

By band, the tail is where the axes separate: frames above 8 deg (21 of them) average
36.8 deg, of which yaw 27.2, roll 13.8, pitch 5.5. Those 21 frames carry 20.4% of the
summed angle over all 1502. The 84 frames the ambiguity gate flags average 10.28 deg
and carry 22.8% — 5.6% of the frames, nearly a quarter of the rotation error.

## 2. `rot_worth.py` — the counterfactual, which reorders the axes

Setting one component of the error to zero, on the same frames:

| removed | MOE | delta |
|---|---|---|
| nothing | 2.5235 | — |
| yaw | 1.6942 | −0.83 |
| roll + pitch | 1.4747 | **−1.05** |
| roll | 2.1166 | −0.41 |
| pitch | 2.0367 | −0.49 |

This is an upper bound (it assumes perfect estimation), but it says something the
squared-error share does not: in MOE terms roll+pitch is worth more than yaw, because
MOE averages the angle while the squared share is dominated by the tail. Both
statements are true and they answer different questions.

## 3. `yaw_ident.py` — is the yaw in the map at all

Park the pose at the **ground-truth translation** and sweep body yaw over +-15 deg.
On the 16 frames with yaw error above 8 deg: **15 are identifiable** (the map prefers
the true yaw and the dip at +-5 deg is sharp, median 0.12 versus 0.20 on control
frames). One is flat/inverted.

That result alone is easy to over-read. It says the information exists where the
pipeline is not.

## 4. `yaw_reach.py` — can the pipeline get there

Same sweep, but parked at the translation the **current model actually outputs**.
This is the decisive version, because a deployment starts there.

| criterion | tail (16 frames) | control (12) |
|---|---|---|
| true yaw inside predicted +-15 deg | 10 | 12 |
| true yaw is a local min at the model's position | 5 | 12 |
| true yaw is the global min at the model's position | 4 | 12 |
| **all three (reachable)** | **5** | **12** |

The gap between 15/16 and 5/16 is the whole point. Three failure modes:

* **outside the window** — 912 (16.7 deg), 1428 (18.3), 1429 (58.0), and three frames
  at ~120 deg; a +-15 deg sweep cannot contain the truth;
* **evidence inverted** — 530 and 917 have the truth inside the window, but at the
  model's position the map prefers a wrong yaw (−12 deg and +8 deg), so even a perfect
  local optimiser would move the wrong way;
* **wrong place entirely** — 1227/1229/1230 have 443 m of translation error. Rotation
  is not the problem there.

Cost of the fix, if it were built: the 5 reachable frames carry about 59 deg of angle
between them, which is roughly 0.04 deg of MOE **if this branch ran alone**. It does
not. In the assembled model 13 of these 16 frames are already gated and 10 are already
replaced by the visual hypothesis (750, 760, 912, 996, 997, 999, 1227, 1229, 1428,
1429), so only 6 frames are left to the along-track branch at all — and 4 of the 5
reachable ones are among the already-adopted. The honest figure is about 0.01 deg, i.e.
one frame (998). A yaw sweep is not a headline improvement; it is a correctness fix for
a branch that the rest of the model already routes around.

What survives is the explanation, not the fix: the along-track branch cannot reach yaw
the map clearly contains, and that is why its own MOE sits where it does. The system
level MOE has to be attacked elsewhere — most plausibly the 670 frames in the 2-8 deg
band (mean 3.08 deg, translations mostly under 1 m), where the control set shows the
map is clean, 12/12. That has to be measured on the assembled model, not on this
branch in isolation.

## Caveats

* The sweeps are local: a +-15 deg window proves a local minimum, never global
  optimality. The +-180 deg sweep in `yaw_reach.py` is there to show when a competing
  basin exists, and for the 120 deg frames it does.
* Axis-angle components are not Euler angles. At large errors the three components
  mix, so the >8 deg band is a magnitude reading.
* `rot_worth.py` numbers are upper bounds, not achievable gains.
* The control set is 12-15 randomly drawn low-error frames; it is a sanity check, not
  a matched comparison group.
