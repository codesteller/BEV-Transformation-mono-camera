#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include "bev/radarcam/validate.hpp"

#include <opencv2/calib3d.hpp>

using Catch::Matchers::WithinAbs;
using bev::radarcam::compute_capture_distribution_diagnostic;
using bev::radarcam::compute_heldout_residuals;
using bev::radarcam::ground_plane_cross_check;
using bev::radarcam::HeldoutInput;
using bev::radarcam::RigidTransform;

namespace {
cv::Matx33d rodrigues(double x, double y, double z) {
    cv::Mat rvec = (cv::Mat_<double>(3, 1) << x, y, z);
    cv::Mat R;
    cv::Rodrigues(rvec, R);
    return cv::Matx33d(R);
}
}  // namespace

TEST_CASE("compute_heldout_residuals matches hand-computed 3D and pixel statistics",
    "[phase10][validate][heldout]") {
    // R=I, t=0 -> 3D residual for point q with observed p = q + offset is exactly |offset|. All
    // points share the same depth (z=10) and only shift laterally in x, so pixel reprojection is
    // an exact, hand-computable linear scaling of the 3D residual: px = fx * dx / depth.
    RigidTransform transform;  // identity
    const cv::Matx33d K(100, 0, 50, 0, 100, 50, 0, 0, 1);

    const std::vector<double> dx = {0.01, 0.02, 0.03, 0.04, 0.05};
    std::vector<HeldoutInput> heldout;
    for (double d : dx) {
        HeldoutInput h;
        h.q_radar = cv::Point3d(0.0, 0.0, 10.0);
        h.p_camera = cv::Point3d(d, 0.0, 10.0);
        heldout.push_back(h);
    }

    const auto stats = compute_heldout_residuals(heldout, transform, K);
    REQUIRE(stats.n_holdout == 5);

    // Hand-computed (not re-derived from the implementation): residuals = [0.01..0.05].
    // RMS = sqrt(mean([1,4,9,16,25]e-4)) = sqrt(0.0011) = 0.0331662...
    CHECK_THAT(stats.rms_3d_m, WithinAbs(0.033166247903554, 1e-9));
    CHECK_THAT(stats.median_3d_m, WithinAbs(0.03, 1e-9));
    // p95, linear interpolation over 5 sorted values: rank=0.95*4=3.8 -> 0.04*0.2 + 0.05*0.8 = 0.048
    CHECK_THAT(stats.p95_3d_m, WithinAbs(0.048, 1e-9));

    // Pixel residuals are exactly 10x the 3D residuals here (fx=100, depth=10 -> factor fx/depth=10).
    CHECK_THAT(stats.rms_reprojection_px, WithinAbs(0.33166247903554, 1e-9));
    CHECK_THAT(stats.median_reprojection_px, WithinAbs(0.3, 1e-9));
    CHECK_THAT(stats.p95_reprojection_px, WithinAbs(0.48, 1e-9));
}

TEST_CASE("compute_heldout_residuals handles an empty holdout set without dividing by zero",
    "[phase10][validate][heldout]") {
    RigidTransform transform;
    const cv::Matx33d K(100, 0, 50, 0, 100, 50, 0, 0, 1);
    const auto stats = compute_heldout_residuals({}, transform, K);
    CHECK(stats.n_holdout == 0);
    CHECK(stats.rms_3d_m == 0.0);
}

TEST_CASE("ground_plane_cross_check recovers a known composed radar<->ground pose",
    "[phase10][validate][ground_check]") {
    const cv::Matx33d K(800, 0, 320, 0, 800, 240, 0, 0, 1);

    // Ground truth camera<->ground pose (world->camera): a modest tilt, arbitrary but fixed.
    const cv::Matx33d R_wc = rodrigues(0.05, 0.1, 0.02);
    const cv::Vec3d t_wc(0.2, -0.1, 3.0);

    // Build the world->pixel homography H = K*[r1 r2 t] directly from R_wc/t_wc's own columns,
    // matching the classical planar decomposition this function is expected to invert.
    const cv::Matx33d M(R_wc(0, 0), R_wc(0, 1), t_wc[0], R_wc(1, 0), R_wc(1, 1), t_wc[1], R_wc(2, 0),
        R_wc(2, 1), t_wc[2]);
    const cv::Matx33d H_world_to_pixel = K * M;
    const cv::Matx33d H_pixel_to_world = H_world_to_pixel.inv();  // what HomographyData actually stores

    bev::HomographyData homography;
    homography.image_width = 640;
    homography.image_height = 480;
    homography.plane_width_m = 2.0;
    homography.plane_height_m = 2.0;
    homography.homography_matrix = cv::Mat(H_pixel_to_world);

    const cv::Matx33d R_rc = rodrigues(0.15, -0.1, 0.05);
    const cv::Vec3d t_rc(0.3, -0.2, 1.0);
    RigidTransform radar_to_camera;
    radar_to_camera.R = R_rc;
    radar_to_camera.t = t_rc;

    // Independently composed expected result (not calling the function under test).
    const cv::Matx33d expected_R_radar_to_ground = R_wc.t() * R_rc;
    const cv::Vec3d expected_t_radar_to_ground = R_wc.t() * (t_rc - t_wc);
    const double expected_height = std::abs(expected_t_radar_to_ground[2]);
    double expected_pitch_deg, expected_roll_deg;
    bev::radarcam::rotation_to_pitch_roll_deg(expected_R_radar_to_ground, expected_pitch_deg, expected_roll_deg);

    const auto result = ground_plane_cross_check(homography, K, radar_to_camera);
    REQUIRE(result.success);
    CHECK_THAT(result.derived_radar_height_m, WithinAbs(expected_height, 1e-6));
    CHECK_THAT(result.radar_pitch_deg, WithinAbs(expected_pitch_deg, 1e-6));
    CHECK_THAT(result.radar_roll_deg, WithinAbs(expected_roll_deg, 1e-6));
}

TEST_CASE("ground_plane_cross_check fails cleanly on a degenerate homography", "[phase10][validate][ground_check]") {
    bev::HomographyData homography;
    homography.homography_matrix = cv::Mat::zeros(3, 3, CV_64F);  // degenerate -- zero columns
    const cv::Matx33d K(800, 0, 320, 0, 800, 240, 0, 0, 1);
    RigidTransform radar_to_camera;

    const auto result = ground_plane_cross_check(homography, K, radar_to_camera);
    CHECK_FALSE(result.success);
    CHECK_FALSE(result.error.empty());
}

TEST_CASE("compute_capture_distribution_diagnostic flags a coplanar cloud, passes a well-distributed one",
    "[phase10][validate][distribution]") {
    std::vector<cv::Point3d> coplanar = {{5, 0, 0}, {5, 2, 0}, {10, -3, 0}, {15, 1, 0}, {8, -4, 0}, {12, 0, 0}};
    const auto coplanar_diag = compute_capture_distribution_diagnostic(coplanar, 0.15);
    REQUIRE(coplanar_diag.valid);
    CHECK(coplanar_diag.degenerate_warning);
    CHECK_THAT(coplanar_diag.height_spread_m, WithinAbs(0.0, 1e-9));
    CHECK(coplanar_diag.height_spread_flagged);

    std::vector<cv::Point3d> well_distributed = {{5, 0, 0}, {5, 2, 0}, {5, -2, 1}, {10, 3, -1}, {10, -3, 2},
        {15, 1, 1.5}, {15, -1, -1.5}, {20, 0, 2}, {8, 4, 0.5}, {8, -4, -0.5}, {12, 0, 3}, {12, 0, -3}};
    const auto good_diag = compute_capture_distribution_diagnostic(well_distributed, 0.15);
    REQUIRE(good_diag.valid);
    CHECK_FALSE(good_diag.degenerate_warning);
    CHECK(good_diag.height_spread_m > 0.3);
    CHECK_FALSE(good_diag.height_spread_flagged);
}

TEST_CASE("compute_capture_distribution_diagnostic is invalid with fewer than 3 points",
    "[phase10][validate][distribution]") {
    std::vector<cv::Point3d> too_few = {{5, 0, 0}, {10, 0, 0}};
    const auto diag = compute_capture_distribution_diagnostic(too_few);
    CHECK_FALSE(diag.valid);
}
