#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include "bev/radarcam/gate_iterate.hpp"

#include <opencv2/calib3d.hpp>

using Catch::Matchers::WithinAbs;
using bev::radarcam::CaptureInput;
using bev::radarcam::ClutterFilterParams;
using bev::radarcam::run_gate_iterate;

namespace {

cv::Matx33d rodrigues(double x, double y, double z) {
    cv::Mat rvec = (cv::Mat_<double>(3, 1) << x, y, z);
    cv::Mat R;
    cv::Rodrigues(rvec, R);
    return cv::Matx33d(R);
}

cv::Point3d apply(const cv::Matx33d& R, const cv::Vec3d& t, const cv::Point3d& q) {
    const cv::Vec3d qv(q.x, q.y, q.z);
    const cv::Vec3d p = R * qv + t;
    return cv::Point3d(p[0], p[1], p[2]);
}

std::vector<cv::Point3d> radar_frame_points() {
    return {{5, 0, 0}, {5, 2, 0}, {5, -2, 1}, {10, 3, -1}, {10, -3, 2}, {15, 1, 1.5}, {15, -1, -1.5},
        {20, 0, 2}, {8, 4, 0.5}, {8, -4, -0.5}, {12, 0, 3}, {12, 0, -3}};
}

// Builds one CaptureInput per radar-frame point: p_camera is the ground-truth-transformed
// observation (as if PnP produced it exactly), and the dwell holds `n_frames` identical
// detections at that same radar-frame point, with gates loosened enough that only the spatial
// gate (the thing under test) meaningfully discriminates.
std::vector<CaptureInput> make_captures(
    const cv::Matx33d& R_true, const cv::Vec3d& t_true, int n_frames = 5, double noise_m = 0.0) {
    std::vector<CaptureInput> captures;
    int idx = 0;
    for (const auto& q_true : radar_frame_points()) {
        // Deterministic per-point "measurement scatter" (not random -- reproducible): real radar
        // detections don't land exactly on the noiseless geometric prediction even under the true
        // extrinsic, since this represents sensor-side scatter, not estimation error.
        const double sign = (idx % 2 == 0) ? 1.0 : -1.0;
        const cv::Point3d q_detected = q_true + cv::Point3d(sign * noise_m, -sign * noise_m, sign * noise_m);
        ++idx;

        CaptureInput cap;
        cap.p_camera = apply(R_true, t_true, q_true);
        cap.has_background = false;
        for (int f = 0; f < n_frames; ++f) {
            bev::radarcam::RadarDetection d;
            d.position = q_detected;
            d.range_m = cv::norm(q_detected);
            d.vel_mps = 0.0;
            d.rcs_dbsm = -20.0;
            cap.dwell.frames.push_back({d});
        }
        captures.push_back(cap);
    }
    return captures;
}

ClutterFilterParams loose_params() {
    ClutterFilterParams params;
    params.sigma_trihedral_dbsm = -20.0;
    params.r_ref_m = 1.0;
    params.rcs_margin_db = 1000.0;          // effectively disables the RCS gate for this test
    params.doppler_threshold_mps = 0.5;
    params.radar_height_above_ground_m = 1000.0;  // effectively disables the ground gate
    params.persistence_frac = 0.0;          // every frame has a detection anyway
    params.spread_max_m = 10.0;             // identical repeated detections -> spread is 0 regardless
    params.background_match_radius_m = 0.3;
    return params;
}
}  // namespace

TEST_CASE("run_gate_iterate converges to the true extrinsic within a few iterations", "[phase8][gate_iterate]") {
    const cv::Matx33d R_true = rodrigues(0.2, -0.1, 0.05);
    const cv::Vec3d t_true(0.3, -0.2, 1.1);

    // Deliberately-off coarse extrinsic: small rotation error (~1 deg) + ~0.37m translation error --
    // large enough that a tight gate would miss real detections on the first pass, small enough
    // that the wide first-pass gate (2.0m) still admits every true correspondence.
    const cv::Matx33d R_coarse = rodrigues(0.21, -0.11, 0.06);
    const cv::Vec3d t_coarse(0.6, -0.4, 1.2);

    const auto captures = make_captures(R_true, t_true);
    const std::vector<double> gate_radii_m = {2.0, 0.4, 0.3};

    const auto result = run_gate_iterate(
        captures, gate_radii_m, R_coarse, t_coarse, loose_params(), /*ransac_inlier_threshold_m=*/0.05, /*inlier_drop_warn_frac=*/0.5);

    REQUIRE(result.success);
    REQUIRE(result.iterations.size() == 3);

    // First (wide) pass should have found (close to) all 12 correspondences as candidates and inliers.
    CHECK(result.iterations[0].n_candidates == 12);
    CHECK(result.iterations[0].n_inliers == 12);

    // By the final (tight) iteration, the improved estimate should still admit all 12 -- the gate
    // didn't collapse, it tightened correctly around a now-accurate prediction.
    CHECK(result.iterations.back().n_candidates == 12);
    CHECK(result.iterations.back().n_inliers == 12);

    for (int r = 0; r < 3; ++r) {
        for (int c = 0; c < 3; ++c) {
            CHECK_THAT(result.transform.R(r, c), WithinAbs(R_true(r, c), 1e-6));
        }
    }
    for (int i = 0; i < 3; ++i) {
        CHECK_THAT(result.transform.t[i], WithinAbs(t_true[i], 1e-6));
    }

    // No iteration should have tripped the guardrail -- inliers never dropped sharply.
    for (const auto& rec : result.iterations) {
        CHECK_FALSE(rec.guardrail_triggered);
    }
}

TEST_CASE("run_gate_iterate's guardrail fires when a later gate radius collapses the inlier set",
    "[phase8][gate_iterate]") {
    const cv::Matx33d R_true = rodrigues(0.2, -0.1, 0.05);
    const cv::Vec3d t_true(0.3, -0.2, 1.1);

    // Genuine (not just floating-point) per-detection scatter of ~0.05m (see make_captures) --
    // realistic sensor-side noise, not estimation error, since the coarse extrinsic here is exact.
    const auto captures = make_captures(R_true, t_true, /*n_frames=*/5, /*noise_m=*/0.03);
    // First pass wide enough to find everything; second pass tighter than the ~0.05m real scatter,
    // so inliers should crash from 12 to (near) 0.
    const std::vector<double> gate_radii_m = {2.0, 0.01};

    // Inlier threshold comfortably covers the ~0.052m noise magnitude above, so the first (wide)
    // pass finds all 12 as inliers -- the second pass's 0.01m *spatial gate* is what should collapse.
    const auto result = run_gate_iterate(
        captures, gate_radii_m, R_true, t_true, loose_params(), /*ransac_inlier_threshold_m=*/0.1, /*inlier_drop_warn_frac=*/0.5);

    REQUIRE(result.iterations.size() >= 1);
    CHECK(result.iterations[0].n_inliers == 12);

    // The second iteration either found too few candidates to solve (n_candidates < 3, loop stops
    // early) or solved with a collapsed inlier set that trips the guardrail -- either way, the run
    // must not silently report success with a healthy-looking second iteration.
    if (result.iterations.size() >= 2) {
        const bool collapsed_badly =
            result.iterations[1].n_candidates < 3 || result.iterations[1].guardrail_triggered;
        CHECK(collapsed_badly);
    }
}
