# VERA: Visual Endpoint Re-Anchoring inside LEADER's correspondence prediction

**Core idea.** Every LEADER correspondence pairs a fixed local endpoint (the LiDAR sample) with a map coordinate that the network predicts. In VERA, the image decides where on the LiDAR-measured surface that endpoint should sit. LiDAR geometry then gives the endpoint's local coordinates exactly, and MMRegressor predicts its map coordinate. VERA runs in one pass, uses no pose prior and stores no map appearance. SC2-PCR, the Tukey/SVD refinement and the final center/transform restoration are unchanged.

## 1. The internal problem: endpoints can't be pinned down on degenerate surfaces

Where a LiDAR sample $p_i$ lands on a surface depends on the beam pattern and the vehicle pose, not on anything in the scene. On surfaces that look the same to LiDAR as you move along them (facades, road, poles), geometry only fixes $T^*p_i$ along the surface normal. Along the surface, the regressor has to carry position over from distinctive structures that may be far away. Its error $e_i=\hat y_i-T^*p_i$ therefore lies along the surface. Because neighboring points share receptive fields, the error is also roughly the same across a whole region. LEADER's TRR is designed to mitigate error propagation from degenerate regions, so these regions are down-weighted rather than resolved.

**Effect on the pose.** Linearize the converged Tukey/SVD solution, with $\hat R=\mathrm{Exp}(\delta\theta)R^*$, $\hat t=t^*+\delta t$, weights $w_i$, $a_i=R^*p_i$, centered points $z_i=a_i-\bar a$ and mean error $\bar e=\sum w_ie_i/\sum w_i$:

$$\delta\theta=H^{-1}\sum_i w_i\,z_i\times e_i,\qquad H=\sum_i w_i\left(\|z_i\|^2I_3-z_iz_i^\top\right),\qquad \delta t=\bar e-\delta\theta\times\bar a \tag{1}$$

- Independent errors average out.
- Region-coherent errors along the surface enter the torque sum directly. Horizontal errors drive yaw and position; vertical errors on walls drive roll and pitch.
- Surfaces that TRR suppresses contribute nothing to $H$.

**Why the backend keeps these errors.** If two correspondences share the same error $e$, then $\|\hat y_i-\hat y_j\|=\|p_i-p_j\|$ exactly, so SC2-PCR's pairwise length check sees them as perfectly consistent.

*Toy example:* facades at $y=\pm10$ m, $x\in[-30,30]$ m, along-facade errors $\pm0.3$ m.
- Eq. (1) gives $\delta\theta_z=-Ds/(X^2/3+D^2)=-3/400$ rad, i.e. a 0.43° yaw error.
- Residuals after the fit are at most 0.32 m, so Tukey weights stay at least 0.52 with $c=0.6$ m.
- Every correspondence survives, and so does the bias.

**From problem to mechanism.** Suppose vision only adds information to the prediction for a fixed $p_i$. The regressor would still have to turn image offsets between the beam hit and nearby visual structure into metric map offsets, which depend on depth and viewing angle. That is again learned propagation. VERA removes that step: it moves the endpoint onto a point the image can recognize, and LiDAR supplies that point's coordinates.

## 2. The module

For each output voxel $i$ (local frame, $f_i\in\mathbb R^{512}$), using the side camera where it projects closest to the principal point:

1. **Local surface patch.** Take raw points within $r=1$ m of $p_i$. Compute $C_i=\mathrm{cov}=\sum_k\lambda_k e_ke_k^\top$ with $\lambda_1\ge\lambda_2\ge\lambda_3$, and set $B_i=[\sqrt{\lambda_1}e_1,\sqrt{\lambda_2}e_2]\in\mathbb R^{3\times2}$.
2. **Candidate points.** $c_{im}=p_i+B_i\eta_m$, with $\eta_m$ on a 7×7 grid over $[-\kappa,\kappa]^2$, $\kappa=\sqrt2$ (49 candidates). Keep only candidates that are visible: positive depth, inside the image, and passing a LiDAR z-buffer test at $x_{im}=\pi_c(c_{im})$ with tolerance 0.3 m. If a candidate is kept, its mirror $-\eta_m$ must be kept too, so the visible set stays symmetric.
3. **Cross-modal attention.**
   $$s_{im}=d^{-1/2}\,(W_qf_i)^\top\big(W_k\,G_c(x_{im})+P\,\gamma(F_i^\top B_i\eta_m)\big),\qquad a_{im}=\mathrm{softmax}_m(s_{im})$$
4. **New endpoint (anchor).** $p_i^a=\sum_m a_{im}c_{im}=p_i+\Delta_i$, where $\Delta_i=B_i\bar\eta_i$ and $\bar\eta_i=\sum_m a_{im}\eta_m$.
5. **Prediction.** Sample $v_i=G_c(\pi_c(p_i^a))$ at the anchor's own pixel. Then $[\hat y_i^a;u_i]=\mathrm{MMReg}\big(f_i+U_vv_i+U_\Delta\gamma(F_i^\top\Delta_i)\big)$.
6. **To the backend.** Pass $(p_i^a,\ \hat y_i^a+$`center_t`$,\ u_i)$. Voxels outside all cameras use $\Delta_i=0$, $v_i=0$.

**Notation.**
- $F_i=[\hat\rho_i,\hat\theta_i,\hat z]$ is the cylindrical frame at the point (radial, azimuthal, up).
- $G_c$ is frozen SAM (stride 16) upsampled 4×, concatenated with a trainable ImageNet ResNet-18 conv1–layer1 (stride 4), then a 1×1 conv to 64 channels. SAM carries appearance identity; the stride-4 stem gives about 1-pixel localization, which check C6 relies on.

**Dimensions.**

| Quantity | Size |
|---|---|
| $f_i$ | 512 |
| $W_q$ | 64×512 |
| $G_c(\cdot)$, $v_i$ | 64 |
| $W_k$ | 64×64 |
| $\gamma$ | Fourier features, ℝ³→ℝ⁴⁸ |
| $P$ | 64×48 |
| $B_i$ | 3×2 |
| $\bar\eta_i$ | ℝ² |
| $\Delta_i$, $p_i^a$ | ℝ³ |
| $U_v$ | 512×64 |
| $U_\Delta$ | 512×48 |
| Output | ℝ⁴ (unchanged) |

**How vision takes part:**
- **Where:** visual keys choose the endpoint within the measured surface patch (steps 3–4).
- **What:** $v_i$ conditions both the coordinate and the reliability (step 5).
- **Which:** $u_i$ is learned on the re-anchored residuals, so the u-ranked pool shifts toward re-anchored correspondences on surfaces that used to be suppressed.

**No gating.** No learned coefficient scales any branch. Vision acts only by choosing a location: the softmax output is a point on the measured surface.

## 3. Consistency checks

**C1 – The anchor stays on the measured surface.**
- $e_3^\top\Delta_i=0$ exactly, so vision can never move the endpoint along the normal, which is the direction LiDAR measures well.
- On poles ($\lambda_2\to0$) the allowed motion collapses to the pole axis.
- $\|\Delta_i\|\le\kappa\sqrt{\lambda_1+\lambda_2}\le\kappa\sqrt{\mathrm{tr}\,C_i}\le\kappa r$. For a disk-shaped patch ($\lambda_{1,2}=r^2/4$) the bound is exactly 1 m, so candidates stay on measured support.
- Grid step is $\kappa\sigma/3\approx0.24$ m, close to the voxel size.

**C2 – Training targets are exact for soft anchors.** The weights sum to 1 and $T^*$ is affine, so $T^*p_i^a=\sum_m a_{im}T^*c_{im}$. The target is exact for any attention pattern. The planarity assumption only affects whether the pixel and the anchor refer to the same physical point, never whether the label is correct.

**C3 – Starts exactly as pretrained LEADER.** With $W_q=0$, attention is uniform over a symmetric set, so $\bar\eta=0$ and $p^a=p$. With $U_v=U_\Delta=0$, the outputs equal pretrained LEADER's.

**C4 – Gradients only let the anchor fix along-surface error.**
- $\partial r_i/\partial\bar\eta_i=\partial\hat y_i^a/\partial\bar\eta_i-R^*B_i$. The geometric term only spans the tangent plane in the map frame.
- So the anchor can absorb only along-surface residual; the regressor must fix any normal residual.
- A LiDAR-only bias can be removed by the regressor at no cost. With the $\beta\|\bar\eta\|^2$ penalty, the anchor therefore stays put ($\Delta=0$) in those cases. A displacement survives only where the image feature at the moved point makes its map coordinate predictable.
- Leakage from the regressor's implicit rotation error: at most $0.0314\text{ rad}\times1\text{ m}=3.1$ cm.

**C5 – Yaw invariance is preserved.** Under a sensor yaw rotation $R_z$: $\Delta\to R_z\Delta$ and $F_i\to R_zF_i$, so $F_i^\top\Delta_i$ is unchanged. The new inputs keep RPGE's yaw-invariant geometric encoder property.

**C6 – Anchor precision depends on direction.** For pixel repeatability $\sigma_{px}$ at range $Z$, near the optical axis, with incidence angle $\varphi$:
- in the incidence plane: $\sigma_\parallel=Z\sigma_{px}/(f\cos\varphi)$
- perpendicular to it: $\sigma_\perp=Z\sigma_{px}/f$

With $f=800$ px and $\sigma_{px}=1$:
- **Road point 20 m ahead of a camera 2 m high:** $\varphi=84.3°$, giving 2.5 cm azimuthally and 25 cm radially. A radial error $e\parallel(z_x,z_y)$ gives $(z\times e)_z=0$. So the foreshortened direction doesn't affect yaw, and the direction that does is precise to 2.5 cm.
- **Facade at 20 m, $\varphi=60°$:** 2.5 cm vertically (matters for roll/pitch), 5 cm horizontally.

**C7 – Effect on the final pose.**
- The torque in (1) is linear in the error. Cutting the along-surface error from $s=0.3$ m to $s'=0.05$ m cuts the toy yaw error from 0.43° to 0.07°.
- Tukey weights rise from 0.56 to 0.986, so re-anchored correspondences dominate the refinement. That is how direction-dependent confidence from vision gets into an isotropic backend: through the coordinates, not the weights.
- Adding accurate far points to $H$ only makes it larger in the positive-semidefinite (PSD) sense, so $H^{-1}$ shrinks.

## 4. Training

- **Stage 1: RPGE frozen.** Train the stem and 1×1 conv, $W_q$, $W_k$, $P$, $U_v$, $U_\Delta$, and all of MMRegressor (starting from pretrained weights). The regressor has to change because its target changes from $T^*p$ to $T^*p^a$. Frozen queries keep the anchor–target co-adaptation in C4 well-posed. SAM stays frozen throughout.
- **Stage 2:** unfreeze RPGE's decoder stages at 0.1× learning rate.
- **Loss:** $\mathcal L=\sum_i\mathrm{TRR}\big(\hat y_i^a,\,T^*p_i^a-\texttt{center\_t};\,u_i\big)+\beta\sum_i\|\bar\eta_i\|^2$. Set $\beta=0.01$, so the largest displacement ($\|\bar\eta\|^2=4$) costs about 4 cm in TRR units. Gradients flow through $T^*p^a$, the softmax and the bilinear sampling, with no stop-gradient.

## 5. Assumptions and conditions

- **Projection accuracy:** calibration plus LiDAR–camera time alignment must be much finer than the 0.24 m grid step. Otherwise visual identity gets attached to the wrong point; the labels stay exact (C2).
- **Local planarity:** the surface must stay close to the fitted plane within the patch, e.g. $d^2/2\rho_c=3.6$ cm for $d=0.6$ m and curvature radius $\rho_c=5$ m. The z-buffer test handles occlusions caused by the LiDAR–camera offset.
- **Stable appearance at anchors** between training and test (carried by the frozen SAM features).
- **Linearization holds:** $|\delta\theta|\ll1$ rad (1.8° = 0.031 rad), and residuals stay inside the Tukey support.
- **Camera coverage on both sides:** in the toy example, re-anchoring only one facade halves the yaw error (0.21°) but creates a 0.15 m position error ($\bar e=0.15$ m). Position and rotation both improve only if anchors exist at opposite azimuths, so all side-facing cameras should be used, not Cam5 alone.