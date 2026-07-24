# Radar–Camera Extrinsic Calibration Tool — Implementation Spec

**Project:** OpenADAS v2.0 (Gahan AI)
**Target:** Offline calibration tool, runs on workstation (RTX 5090 / 256GB RAM), Python 3.10+
**Scope:** 6-DOF extrinsic calibration between front 4D radar (range/azimuth/elevation/Doppler) and front camera, using a single trihedral corner reflector rigidly mounted to a checkerboard target.

---

## 1. Goal

Solve for the rigid transform `(R, t)` mapping points from the **radar frame** into the **camera frame**:

```
X_camera = R · X_radar + t
```

`R` is 3×3 (SO(3)), `t` is 3×1. Six degrees of freedom. Both sensors are rigidly mounted to the vehicle, so this is a single fixed transform — **not** solved per-frame.

The 4D radar resolves elevation, so all 6 DOF are observable. (A separate reduced-DOF path for the 3D radar is out of scope for v1 — see §11.)

---

## 2. Physical setup and assumptions

- **Target rig:** one trihedral corner reflector **rigidly bolted** to a checkerboard. The reflector's position in board coordinates, `X_B` (3×1), is constant across all captures and measured once with calipers.
- `X_B` must account for the **phase-center standoff** — the trihedral's radar phase center is near its apex, not the board face.
- Board is **radar-transparent** (foam/cardboard, printed checkerboard). Mount is non-metallic. The reflector must be the dominant scatterer on the rig.
- **Camera intrinsics `K` and distortion coefficients are already calibrated** and supplied as input. This tool does not solve intrinsics.
- Vehicle stationary, rig stationary during each capture (stop-and-capture, not drive-by). This sidesteps sensor time-sync entirely.
- A **coarse extrinsic** (tape measure + inclinometer, ±20cm / ±5°) is supplied as a config input to bootstrap the spatial gate.

---

## 3. Data model

### 3.1 Capture record (on disk, one directory per capture)

```
capture_0001/
  image.png                 # camera frame, target present
  radar_dwell.npy           # (F, D, 7) — F frames in dwell, D detections
  radar_bg.npy              # optional: dwell with reflector removed/rotated
  meta.json                 # rig height, timestamp, operator notes
```

Radar detection columns: `[range_m, azimuth_rad, elevation_rad, doppler_mps, rcs_dbsm, snr_db, valid]`

### 3.2 Core dataclasses

```python
@dataclass
class RadarDetection:
    range_m: float
    azimuth_rad: float
    elevation_rad: float
    doppler_mps: float
    rcs_dbsm: float
    snr_db: float

    def to_cartesian(self) -> np.ndarray:
        """Radar spherical -> Cartesian. x fwd, y left, z up."""

@dataclass
class Capture:
    capture_id: str
    image_path: Path
    radar_dwell: np.ndarray          # (F, D, 7)
    radar_background: np.ndarray | None
    rig_height_m: float | None

@dataclass
class Correspondence:
    capture_id: str
    p_camera: np.ndarray             # (3,) reflector position, camera frame (from PnP)
    q_radar: np.ndarray              # (3,) reflector position, radar frame (from detection)
    board_pose: tuple[np.ndarray, np.ndarray]   # (R_k, t_k) retained for §8 refinement
    dwell_spread_m: float            # QC metric
    detection_rcs_dbsm: float

@dataclass
class CalibrationResult:
    R: np.ndarray                    # (3,3)
    t: np.ndarray                    # (3,)
    n_inliers: int
    n_total: int
    rms_residual_m: float
    rms_reprojection_px: float
    euler_deg: tuple[float, float, float]
```

### 3.3 Coordinate conventions (fix these early, document in code)

- **Radar frame:** x forward, y left, z up. Azimuth θ measured from +x toward +y. Elevation φ from the horizontal plane, positive up.
- **Camera frame:** OpenCV convention — x right, y down, z forward.
- **Board frame:** origin at first detected checkerboard corner, x/y in the board plane, z out of the board face toward the camera.

```
x = r · cos(φ) · cos(θ)
y = r · cos(φ) · sin(θ)
z = r · sin(φ)
```

---

## 4. Module architecture

Clean separation of concerns; each module independently testable with synthetic data.

```
radar_cam_calib/
  __init__.py
  config.py           # Pydantic config schema + YAML loading
  io/
    capture_loader.py # discover + load capture dirs
    schemas.py        # dataclasses above
  camera/
    board_detect.py   # checkerboard detection (subpixel)
    pnp.py            # board pose -> reflector position in camera frame
  radar/
    geometry.py       # spherical <-> Cartesian
    clutter.py        # multi-stage clutter filter (§6)
    dwell.py          # persistence check + median aggregation
  solve/
    kabsch.py         # closed-form Umeyama with determinant correction
    ransac.py         # RANSAC wrapper around kabsch
    refine.py         # nonlinear refinement, optional joint X_B estimation
    iterate.py        # gate -> solve -> re-gate loop (§7)
  validate/
    holdout.py        # train/test split, residuals, reprojection error
    ground_check.py   # homography-based independent cross-check (§9)
  report/
    plots.py          # diagnostic figures
    funnel.py         # per-stage detection counts
  cli.py              # entry points
```

**Style:** PEP 8, Black, 4-space indent. Full docstrings on every public function including the math being implemented and units of every argument. NumPy for linear algebra, OpenCV for board detection and PnP, SciPy for nonlinear refinement.

---

## 5. Camera-side pipeline

Per capture:

1. Load image, detect checkerboard corners (`cv2.findChessboardCorners` + `cv2.cornerSubPix`).
2. Solve PnP against known board geometry and supplied intrinsics → `(R_k, t_k)` = board pose in camera frame. Use `cv2.SOLVEPNP_IPPE` (planar target) with iterative refinement.
3. Predict reflector position in camera frame:

```
p_k = R_k · X_B + t_k
```

**Quality gates — reject the capture if:**
- Board not fully detected
- PnP reprojection RMS > threshold (default 1.0 px)
- Board viewing angle is near-frontal (< ~15° tilt) → poorly conditioned pose

Log the board's orientation per capture; §8 depends on having orientation diversity.

---

## 6. Radar clutter filtering (multi-stage, ordered cheap→expensive)

Implement as a pipeline of composable filter stages, each recording how many detections entered and survived.

### Stage 0 — Background subtraction (strongest, do first)
If `radar_bg.npy` is present, match detections between target-present and target-absent dwells by proximity in (range, azimuth, elevation) and remove matches. What survives is the target by construction. Falls back gracefully to a single session-wide background scan if per-capture backgrounds are unavailable.

### Stage 1 — Spatial gate (bootstrap)
Using the current extrinsic estimate, predict the reflector's expected radar-frame position:

```
q̂_k = R_est⁻¹ · (p_k − t_est)
```

Accept detections within `gate_radius_m` of `q̂_k`. Radius is **iteration-dependent**: start ~2.0m on the coarse extrinsic, tighten to ~0.3–0.5m after the first solve (§7).

### Stage 2 — Doppler gate
Target and vehicle are static → reflector sits at zero Doppler. Reject `|doppler| > doppler_threshold` (default 0.5 m/s). Kills moving clutter: passing vehicles, the operator, wind-blown foliage.

> **Operational note to surface in docs:** whoever places the rig must walk out of the radar FOV before the dwell starts.

### Stage 3 — Range-adaptive RCS gate
Received power falls as **1/r⁴** — a fixed amplitude threshold will reject every distant detection. Predict expected return from known trihedral RCS and predicted range `r̂`:

```
P_expected_dbsm ≈ σ_trihedral_dbsm − 40·log10(r̂ / r_ref)
accept if |P_measured − P_expected| < rcs_margin_db     # default ±10 dB
```

Two-sided on purpose: a detection *far brighter* than expected is probably a flat plate or environmental corner, not the reflector.

### Stage 4 — Physical plausibility
Reject any detection whose reconstructed `z` is below the ground plane. Ground multipath produces a ghost at longer range with mirrored (negative) elevation — this is a free multipath filter.

### Stage 5 — Best-in-gate
Take the maximum-RCS survivor. This is what defeats **antenna sidelobes**, which appear at the same range and zero Doppler, displaced in azimuth, at lower amplitude. They pass every other test. A naive "closest to prediction" rule can select one; amplitude is the discriminant that separates mainlobe from sidelobe.

### Stage 6 — Dwell persistence + aggregation
Across the F frames of the dwell:

```
accept if detected in > persistence_frac of frames      # default 0.8
     and positional spread across dwell < spread_max_m  # default 0.10
then q_k = median(positions)
```

Median, not mean — robust to occasional flicker. The spread value is retained on the `Correspondence` as a QC metric; large spread on a *static* dwell indicates a bad capture that should be dropped rather than fed to the solver.

### Funnel logging (required)
Record per-capture detection counts entering and leaving every stage. When a capture yields zero correspondences you must be able to tell *which stage ate it* — empty gate (bad coarse extrinsic in that region), failed persistence, or an over-aggressive RCS cut at range.

---

## 7. Solver

### 7.1 Kabsch / Umeyama (closed form)

Given index-matched clouds `{p_i}` (camera) and `{q_i}` (radar), minimize `Σ ‖p_i − (R·q_i + t)‖²`:

```
1. centroid_q = mean(q);  centroid_p = mean(p)
2. Q = q − centroid_q;    P = p − centroid_p          # centering decouples R from t
3. H = Qᵀ P                                            # 3×3 cross-covariance
4. U, S, Vt = svd(H);  V = Vt.T
5. d = sign(det(V @ U.T))
   R = V @ diag(1, 1, d) @ U.T                         # determinant fix: forbid reflections
6. t = centroid_p − R @ centroid_q
```

**The determinant correction is mandatory, not optional.** `V·Uᵀ` is orthogonal but may be a reflection (det = −1), which is physically meaningless for a rigidly mounted sensor. The `diag(1,1,d)` absorbs the flip into the smallest-singular-value direction — the axis where the data has least spread and is therefore least trustworthy. Add a unit test that feeds near-coplanar data and asserts `det(R) ≈ +1`.

Requires ≥3 non-collinear correspondences. Globally optimal, no initialization, no iteration.

### 7.2 RANSAC wrapper

Sample 3 correspondences → fit → count inliers within `inlier_threshold_m` (start 0.25m) → keep best consensus → refit on all inliers.

> RANSAC is the **last** line of defense, not the first. It rejects a handful of bad pairs among many good ones, but if a large fraction of correspondences are clutter it can lock onto a self-consistent wrong solution. Every filter in §6 exists so RANSAC only cleans up stragglers.

### 7.3 Iterative gate tightening

```
1. Gate wide (~2.0 m) using the coarse tape-measure extrinsic
2. Solve Kabsch + RANSAC
3. Re-gate all captures with the improved extrinsic, now tight (~0.3–0.5 m)
4. Re-solve. Repeat 2–3 passes.
```

Converges quickly. Recovers captures a permanently-wide gate would have polluted, and tightens enough to exclude sidelobes that a 2m gate admits.

**Guardrail — implement this check:** track inlier count across iterations. If it drops sharply as the gate tightens, the gate is over-constrained and is now selecting detections that merely *agree with the current estimate*, quietly biasing the solve toward what you already believed. Emit a warning. Keep the final gate at a few times the expected residual, never at it.

---

## 8. Nonlinear refinement (second pass)

Seed from the Kabsch result. Two modes:

**Mode A — refine (R, t) only.** Parametrize R as a rotation vector (Rodrigues, 3 params) so the SO(3) constraint is structural rather than enforced. Minimize with `scipy.optimize.least_squares`, Huber loss. Optionally weight residuals by per-sensor uncertainty — radar is excellent in range (cm) and mediocre in angle (degrees); camera PnP is excellent in angle and weak in depth, with depth error growing with distance. Plain Kabsch weights all axes equally and ignores this.

**Mode B — jointly estimate `X_B` (9 parameters: 3 rotation + 3 translation + 3 for `X_B`).**

This exists because **an error in the measured reflector offset is the one systematic error the residuals cannot detect.** The predicted camera point carries the error as `R_k · e`, rotated by that frame's board pose:

- If board orientation is roughly **constant** across captures, the error rotates identically every frame and becomes a **pure translation bias** in `t`. Reprojection error will look excellent while `t` is silently wrong by centimeters.
- If board orientation **varies**, the error points differently each frame, partially averages out of `t`, and leaks a small rotational bias into `R`.

With 30+ captures spanning varied board orientations there are ample constraints to support 3 extra parameters. Run Mode B as a diagnostic regardless and **report the drift between the estimated `X_B` and the tape-measure value** — a large drift means the physical measurement is wrong.

---

## 9. Validation (two independent axes — both required)

### 9.1 Held-out residuals
Split ~20% of captures out of the solve. On the held-out set report:
- 3D residual `‖p_i − (R·q_i + t)‖` — RMS, median, p95, in meters
- Pixel reprojection error: project `R·q_i + t` through `K` and compare against the PnP-predicted reflector image position

### 9.2 Ground-plane cross-check (catches what §9.1 cannot)
The existing camera pipeline has a **4-point ground homography**. Decompose `H = K·[r₁ r₂ t]` (scale fixed by the four known metric points, `r₃ = r₁ × r₂`) to recover the camera's height and pitch/roll relative to the road surface.

Compose: `camera↔ground` (homography) ∘ `camera↔radar` (solved) ⟹ **radar↔ground**. Compare the *derived* radar mounting height and pitch against the physical tape-measure/inclinometer values.

**This is the check that catches a constant translation bias from a bad `X_B`** — a solve can produce excellent reprojection error while `t` is systematically offset. An independent measurement of radar height is the only thing that exposes it. Agreement within a few cm and a fraction of a degree is the green light.

> **Important scope note for the implementer:** the homography is a *validator only*. It must never be used as a position source in the calibration solve. Ground-placed reflectors are all coplanar at z=0, and feeding a coplanar target set into Kabsch reintroduces exactly the vertical degeneracy the 4D radar was chosen to avoid. The checkerboard exists precisely so targets can be lifted off the ground to varied heights while still recovering full 3D position via PnP.

### 9.3 Capture-distribution diagnostic (run *before* solving)
The conditioning of the solve depends on the 3D spread of **reflector positions**, not on board tilt. A classic failure: tripod at fixed height, walked around the lot near/far/left/right — 40 beautiful captures, all reflector positions coplanar, pitch/roll/tz garbage despite a 4D radar that could have resolved them.

Compute the singular values of the centered reflector position cloud. **Warn loudly if the smallest is a small fraction of the largest** (suggested threshold: < 0.15). Report spread in range, azimuth, and height separately, and flag height specifically — it is the one operators forget.

---

## 10. Capture protocol (encode as validation + document in README)

- **30–50 captures**, rig static during each
- **Range:** ~5m to 40m+
- **Azimuth:** full radar FOV, left edge to right edge
- **Height:** ≥3 distinct levels, ~0.5m to 2.5m — enforce, do not assume
- Keep the reflector **aimed at the radar** throughout; a trihedral's RCS falls off boresight (±20–40°). Get pose diversity by moving the rig around the FOV, not by extreme board tilts. Board tilt serves PnP conditioning only; 20–45° is the useful band.
- 20–50 radar frames per dwell
- Log everything raw — image, board pose, full dwell (not just the median), rig height. If a systematic error surfaces later you want to re-run analysis without re-running the session.

---

## 11. Out of scope for v1 (design for it, don't build it)

The **3D front radar** (no elevation) requires a **reduced 4-DOF** solve — yaw + tx + ty from the data, with roll/pitch/tz pinned externally, because coplanar radar returns make the out-of-plane parameters unobservable. Keep the solver interface general enough to accept a DOF mask, but do not implement it yet.

Note that both front radars observe the same reflector in the same captures, so **capture both radars' frames from the start even while only solving the 4D**. That single session then also yields the 4D↔3D radar transform essentially for free.

---

## 12. Deliverables

1. `calibrate` CLI: `calibrate run --captures ./data --config cfg.yaml --out result.json`
2. `calibrate validate --result result.json --captures ./data` — held-out + ground-check report
3. `calibrate inspect --captures ./data` — pre-solve distribution diagnostic (§9.3) and funnel report
4. YAML config for all thresholds (never hard-code): gate radii per iteration, Doppler threshold, RCS margin, persistence fraction, spread max, inlier threshold, trihedral RCS, coarse extrinsic
5. Output JSON: `R`, `t`, Euler angles (deg), inlier count/total, RMS residual (m), RMS reprojection (px), per-stage funnel counts, estimated vs. measured `X_B`
6. Diagnostic plots: reflector position cloud in 3D (colored by inlier/outlier), residual vs. range, residual vs. azimuth, reprojection overlay on sample images, inlier count vs. gate iteration
7. Unit tests on synthetic data with known ground-truth `(R, t)`: exact recovery in the noiseless case, graceful degradation with noise, correct determinant handling on near-coplanar input, RANSAC rejecting injected outliers