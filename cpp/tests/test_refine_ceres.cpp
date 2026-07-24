#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include "bev/radarcam/refine_ceres.hpp"

#include <cmath>

using Catch::Matchers::WithinAbs;
using bev::radarcam::RefinementInput;
using bev::radarcam::RigidTransform;
using bev::radarcam::refine_mode_a;
using bev::radarcam::refine_mode_b;
using bev::radarcam::solve_kabsch;

namespace {

cv::Matx33d rodrigues(double x, double y, double z) {
    cv::Mat rvec = (cv::Mat_<double>(3, 1) << x, y, z);
    cv::Mat R;
    cv::Rodrigues(rvec, R);
    return cv::Matx33d(R);
}

cv::Point3d apply(const cv::Matx33d& R, const cv::Vec3d& t, const cv::Vec3d& x) {
    const cv::Vec3d p = R * x + t;
    return cv::Point3d(p[0], p[1], p[2]);
}

double vec_norm(const cv::Vec3d& v) { return std::sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]); }

double rotation_angle_deg(const cv::Matx33d& a, const cv::Matx33d& b) {
    // Angle of the relative rotation a^T * b, via its rotation-vector norm.
    const cv::Matx33d rel = a.t() * b;
    cv::Mat rvec;
    cv::Rodrigues(cv::Mat(rel), rvec);
    return cv::norm(rvec) * 180.0 / 3.14159265358979323846;
}

// One board-pose rotation per capture index, spanning a wide range of axes/angles so the
// reflector-offset error (see the Mode B tests) gets rotated differently at every capture --
// exactly the "board orientation varies" condition from spec §8.
cv::Matx33d varied_board_rotation(int i) {
    const double angle_deg = -35.0 + 70.0 * static_cast<double>(i) / 19.0;
    const double angle_rad = angle_deg * 3.14159265358979323846 / 180.0;
    switch (i % 3) {
        case 0:
            return rodrigues(angle_rad, 0.0, 0.0);
        case 1:
            return rodrigues(0.0, angle_rad, 0.0);
        default:
            return rodrigues(0.0, 0.0, angle_rad);
    }
}

cv::Vec3d board_translation(int i) {
    const double z = 5.0 + 30.0 * static_cast<double>(i) / 19.0;  // 5..35 m range diversity
    const double x = 2.0 * std::sin(static_cast<double>(i));
    const double y = 1.0 * std::cos(static_cast<double>(i));
    return cv::Vec3d(x, y, z);
}

}  // namespace

TEST_CASE("refine_mode_a converges to the true (R,t) from a perturbed seed (noiseless)",
    "[phase9][refine_ceres]") {
    const cv::Matx33d R_true = rodrigues(0.2, -0.1, 0.05);
    const cv::Vec3d t_true(0.3, -0.2, 1.1);
    const cv::Vec3d x_b_true(0.0, 0.0, 0.12);

    std::vector<RefinementInput> correspondences;
    for (int i = 0; i < 20; ++i) {
        const cv::Matx33d R_k = varied_board_rotation(i);
        const cv::Vec3d t_k = board_translation(i);
        const cv::Point3d p_camera = apply(R_k, t_k, x_b_true);  // correct X_B -- no injected error
        const cv::Vec3d q_v = R_true.inv() * (cv::Vec3d(p_camera.x, p_camera.y, p_camera.z) - t_true);

        RefinementInput c;
        c.p_camera = p_camera;
        c.q_radar = cv::Point3d(q_v[0], q_v[1], q_v[2]);
        c.board_pose.R = R_k;
        c.board_pose.t = t_k;
        correspondences.push_back(c);
    }

    // Deliberately-perturbed seed (not the true value, not even the Kabsch closed-form answer).
    RigidTransform seed;
    seed.R = rodrigues(0.25, -0.05, 0.1);
    seed.t = cv::Vec3d(0.5, -0.5, 1.3);

    const auto result = refine_mode_a(correspondences, seed, /*huber_delta_m=*/0.1);
    REQUIRE(result.success);

    for (int r = 0; r < 3; ++r) {
        for (int c = 0; c < 3; ++c) {
            CHECK_THAT(result.transform.R(r, c), WithinAbs(R_true(r, c), 1e-5));
        }
    }
    for (int i = 0; i < 3; ++i) {
        CHECK_THAT(result.transform.t[i], WithinAbs(t_true[i], 1e-5));
    }
}

TEST_CASE("constant board orientation: Mode A hides a bad X_B as a deceptively-clean t bias",
    "[phase9][refine_ceres]") {
    const cv::Matx33d R_true = rodrigues(0.15, -0.1, 0.05);
    const cv::Vec3d t_true(0.2, -0.1, 1.0);
    const cv::Vec3d x_b_true(0.0, 0.0, 0.12);
    const cv::Vec3d x_b_wrong(0.03, -0.02, 0.12);  // a plausible few-cm caliper mis-measurement

    const cv::Matx33d R_board_const = rodrigues(0.4, 0.0, 0.0);  // SAME board rotation every capture

    std::vector<RefinementInput> correspondences;
    for (int i = 0; i < 20; ++i) {
        const cv::Vec3d t_k = board_translation(i);  // position still varies -- only rotation is fixed

        // Physical reality: the reflector is actually at R_board_const*x_b_true + t_k.
        const cv::Point3d p_true = apply(R_board_const, t_k, x_b_true);
        const cv::Vec3d q_v = R_true.inv() * (cv::Vec3d(p_true.x, p_true.y, p_true.z) - t_true);

        // What the pipeline actually feeds the solver: computed with the operator's WRONG X_B.
        const cv::Point3d p_observed = apply(R_board_const, t_k, x_b_wrong);

        RefinementInput c;
        c.p_camera = p_observed;
        c.q_radar = cv::Point3d(q_v[0], q_v[1], q_v[2]);
        c.board_pose.R = R_board_const;
        c.board_pose.t = t_k;
        correspondences.push_back(c);
    }

    RigidTransform seed;
    seed.R = R_true;
    seed.t = t_true;
    const auto result = refine_mode_a(correspondences, seed, /*huber_delta_m=*/0.1);
    REQUIRE(result.success);

    // Residuals look deceptively good: the constant rotated error R_board_const*(x_b_wrong -
    // x_b_true) is fully absorbed into t, so the fit is essentially exact.
    double max_residual = 0.0;
    for (const auto& c : correspondences) {
        const cv::Vec3d qv(c.q_radar.x, c.q_radar.y, c.q_radar.z);
        const cv::Vec3d predicted = result.transform.R * qv + result.transform.t;
        const cv::Point3d predicted_pt(predicted[0], predicted[1], predicted[2]);
        max_residual = std::max(max_residual, cv::norm(predicted_pt - c.p_camera));
    }
    CHECK(max_residual < 1e-6);

    // Rotation is essentially untouched by the error...
    CHECK(rotation_angle_deg(result.transform.R, R_true) < 0.1);

    // ...but t is measurably, systematically biased -- exactly the failure mode §8 warns about.
    const cv::Vec3d t_error = result.transform.t - t_true;
    CHECK(vec_norm(t_error) > 0.02);
}

TEST_CASE("varied board orientation: Mode B recovers the true X_B and reduces the t bias Mode A hides",
    "[phase9][refine_ceres]") {
    const cv::Matx33d R_true = rodrigues(0.15, -0.1, 0.05);
    const cv::Vec3d t_true(0.2, -0.1, 1.0);
    const cv::Vec3d x_b_true(0.0, 0.0, 0.12);
    const cv::Vec3d x_b_wrong(0.03, -0.02, 0.12);

    std::vector<RefinementInput> correspondences;
    for (int i = 0; i < 20; ++i) {
        const cv::Matx33d R_k = varied_board_rotation(i);  // DIFFERENT rotation every capture
        const cv::Vec3d t_k = board_translation(i);

        const cv::Point3d p_true = apply(R_k, t_k, x_b_true);
        const cv::Vec3d q_v = R_true.inv() * (cv::Vec3d(p_true.x, p_true.y, p_true.z) - t_true);
        const cv::Point3d p_observed = apply(R_k, t_k, x_b_wrong);

        RefinementInput c;
        c.p_camera = p_observed;
        c.q_radar = cv::Point3d(q_v[0], q_v[1], q_v[2]);
        c.board_pose.R = R_k;
        c.board_pose.t = t_k;
        correspondences.push_back(c);
    }

    RigidTransform seed;
    seed.R = R_true;
    seed.t = t_true;

    const auto mode_a = refine_mode_a(correspondences, seed, /*huber_delta_m=*/0.1);
    REQUIRE(mode_a.success);
    const double mode_a_t_error = vec_norm(mode_a.transform.t - t_true);

    // Mode B, seeded from the operator's (wrong) measured X_B -- since that's the only value
    // actually available in practice; Mode B's job is to correct it.
    const auto mode_b = refine_mode_b(correspondences, seed, x_b_wrong, /*huber_delta_m=*/0.1);
    REQUIRE(mode_b.success);
    const double mode_b_t_error = vec_norm(mode_b.transform.t - t_true);
    const double x_b_drift_from_true = vec_norm(mode_b.x_b_estimated - x_b_true);
    const double x_b_wrong_distance_from_true = vec_norm(x_b_wrong - x_b_true);

    // Mode B's estimated X_B should land much closer to the TRUE value than the wrong seed was.
    CHECK(x_b_drift_from_true < 0.3 * x_b_wrong_distance_from_true);
    // And Mode B's t bias should be substantially smaller than Mode A's.
    CHECK(mode_b_t_error < 0.3 * mode_a_t_error);
}
