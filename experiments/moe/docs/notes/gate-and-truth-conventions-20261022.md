# Can the correction be made harmless? (2026-10-22)

Two questions: (1) why the remembered V7 number is about 0.16, (2) whether a
parameter change can stop the correction from occasionally making a frame worse.

## 1. The 0.16 and the 0.18 are the same run under two conventions

The V7 report scores against a reference in which the 255 frames of the Feb-12
suspect window (1329071302733069 .. 1329071353938079) are replaced by the official
100 Hz odometry, aligned by one fixed SE3 at the window's anchor. The raw evaluator
truth keeps the deformed poses. Our full online run, rescored with that same policy
module and policy file:

| truth | MPE | MOE | ok(1 m, 2 deg) | >1 m | >5 m |
|---|---|---|---|---|---|
| raw evaluator truth | 0.1843 | 1.4018 | 80.6% | 366 | 284 |
| replaced reference | 0.1660 | 1.4019 | 80.6% | 268 | 29 |

The 255 replaced frames carry about 8 m of error each under the raw truth, which is
255 x 8 / 104728 = 0.019 m of the mean; removing that accounts for the whole gap.
V7 under the same policy is 0.1692 / 1.4053 on 104732 frames, so the current chain is
3.2 mm and 0.0034 deg better, not worse.

## 2. Suppressing the damage costs success rate

A correction is only accepted when the searched pose lowers the truncated cost by
more than 1 cm. That test compares the pose to the map, and the map's own bias is of
the same order as the correction, so a fraction of accepted corrections move the
pose away from truth. Gate on how large the correction is:

| rule | truth | MPE | MOE | ok% | >1 m | worse by >0.1 m | better by >0.1 m |
|---|---|---|---|---|---|---|---|
| cost drop >1 cm (default) | raw | 0.1843 | 1.4018 | 80.6 | 366 | 2505 | 2048 |
| | replaced | 0.1660 | 1.4019 | 80.6 | 268 | 2504 | 2058 |
| correction > 0.5 m | raw | 0.1826 | 1.4033 | 80.4 | 366 | 476 | 716 |
| correction > 0.75 m | raw | 0.1821 | 1.4040 | 80.4 | 381 | 113 | 519 |
| >0.75 m or cost drop >5 cm | raw | 0.1819 | 1.4039 | 80.4 | 381 | 131 | 586 |
| correction > 0.75 m | replaced | 0.1638 | 1.4041 | 80.4 | 290 | 113 | 519 |
| >0.75 m or cost drop >5 cm | replaced | 0.1636 | 1.4039 | 80.4 | 290 | 131 | 586 |

Damage falls by a factor of 20 while the mean position error improves by 2 mm, but
the success rate loses 0.2 points and thirteen more frames end above 1 m. The two
gated rows also lose 0.002 deg of mean orientation error.

The loss comes from the rotation block. Its refinement is fitted on the refined pose
and is only valid together with that pose's translation; applying the rotation alone
to the unrefined pose is worse than not applying it at all (1.4060 against 1.4047), so
the rotation gain cannot be kept while the translation is dropped.

Complete suppression is not reachable this way. The two populations cannot be
separated online: a 0.3 m correction that lands on truth and a 0.3 m correction driven
by map bias look identical in cost. Every threshold that removes the second also
removes the first.

## 3. Recommendation

Keep the current default for the reported numbers; it maximises success rate at
0.1843 / 1.4018 raw and 0.1660 / 1.4019 under the reference protocol, and beats V7 on
both. Report the 0.75 m gate as an ablation: it confines damage to 0.1% of frames at
a cost of 0.2 points of success rate, which is the honest statement of how much the
correction can be trusted.
