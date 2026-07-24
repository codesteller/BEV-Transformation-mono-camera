#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include "bev/radarcam/board_pnp.hpp"

using Catch::Matchers::WithinAbs;
using bev::radarcam::board_object_points;
using bev::radarcam::solve_board_pnp_from_corners;

namespace {
constexpr double kPi = 3.14159265358979323846;

// Rotation about the camera-frame x-axis by `deg` degrees, matching the tilt-angle derivation in
// board_pnp.hpp's comments: with the board's local +z defined as "out of the board face toward the
// camera" (spec §3.3), choosing phi = 180 - tilt_deg here makes solve_board_pnp_from_corners's own
// tilt_deg computation come out to exactly `deg` for the resulting pose -- verified algebraically
// (R*(0,0,1) = (0, -sin(phi), cos(phi)); dot with (0,0,-1) = -cos(phi) = cos(deg) when phi=180-deg).
cv::Matx33d rx_for_tilt_deg(double tilt_deg) {
    const double phi = (180.0 - tilt_deg) * kPi / 180.0;
    const double c = std::cos(phi);
    const double s = std::sin(phi);
    return cv::Matx33d(1, 0, 0, 0, c, -s, 0, s, c);
}

struct SyntheticScene {
    cv::Matx33d K = cv::Matx33d(800, 0, 320, 0, 800, 240, 0, 0, 1);
    int nx = 7;
    int ny = 5;
    double square_m = 0.03;
    cv::Vec3d x_b{0.02, -0.01, 0.05};  // reflector offset in board frame
};

std::vector<cv::Point2f> project_synthetic_corners(
    const SyntheticScene& scene, const cv::Matx33d& R_true, const cv::Vec3d& t_true) {
    const auto object_points = board_object_points(scene.nx, scene.ny, scene.square_m);
    cv::Mat rvec;
    cv::Rodrigues(cv::Mat(R_true), rvec);
    const cv::Mat tvec = (cv::Mat_<double>(3, 1) << t_true[0], t_true[1], t_true[2]);
    const cv::Mat K_mat(scene.K);
    const cv::Mat zero_dist = cv::Mat::zeros(4, 1, CV_64F);

    std::vector<cv::Point2f> corners;
    cv::projectPoints(object_points, rvec, tvec, K_mat, zero_dist, corners);
    return corners;
}
}  // namespace

TEST_CASE("solve_board_pnp_from_corners exactly recovers a known oblique pose (noiseless)",
    "[phase6][board_pnp]") {
    SyntheticScene scene;
    const cv::Matx33d R_true = rx_for_tilt_deg(40.0);
    const cv::Vec3d t_true(0.0, 0.0, 1.3);

    const auto corners = project_synthetic_corners(scene, R_true, t_true);
    const auto result = solve_board_pnp_from_corners(
        corners, scene.nx, scene.ny, scene.square_m, scene.K, scene.x_b, 1.0, 15.0);

    REQUIRE(result.board_found);
    REQUIRE(result.accepted);
    CHECK_THAT(result.pose.reprojection_rms_px, WithinAbs(0.0, 1e-4));
    CHECK_THAT(result.pose.tilt_deg, WithinAbs(40.0, 1e-3));

    for (int r = 0; r < 3; ++r) {
        for (int c = 0; c < 3; ++c) {
            CHECK_THAT(result.pose.R(r, c), WithinAbs(R_true(r, c), 1e-6));
        }
    }
    for (int i = 0; i < 3; ++i) {
        CHECK_THAT(result.pose.t[i], WithinAbs(t_true[i], 1e-6));
    }

    const cv::Vec3d expected_p = R_true * scene.x_b + t_true;
    CHECK_THAT(result.p_camera.x, WithinAbs(expected_p[0], 1e-6));
    CHECK_THAT(result.p_camera.y, WithinAbs(expected_p[1], 1e-6));
    CHECK_THAT(result.p_camera.z, WithinAbs(expected_p[2], 1e-6));
}

TEST_CASE("solve_board_pnp_from_corners degrades gracefully under pixel noise",
    "[phase6][board_pnp]") {
    SyntheticScene scene;
    const cv::Matx33d R_true = rx_for_tilt_deg(35.0);
    const cv::Vec3d t_true(0.05, -0.03, 1.2);

    auto corners = project_synthetic_corners(scene, R_true, t_true);

    // Small, deterministic per-corner perturbation (not random -- keeps the test reproducible)
    // rather than exact noise; large enough to move the reprojection RMS off zero, small enough to
    // stay well under the default 1.0 px gate.
    for (size_t i = 0; i < corners.size(); ++i) {
        const float sign = (i % 2 == 0) ? 1.0f : -1.0f;
        corners[i].x += sign * 0.2f;
        corners[i].y += -sign * 0.15f;
    }

    const auto result = solve_board_pnp_from_corners(
        corners, scene.nx, scene.ny, scene.square_m, scene.K, scene.x_b, 1.0, 15.0);

    REQUIRE(result.board_found);
    REQUIRE(result.accepted);
    CHECK(result.pose.reprojection_rms_px > 0.0);
    CHECK(result.pose.reprojection_rms_px < 1.0);

    const cv::Vec3d expected_p = R_true * scene.x_b + t_true;
    // Bounded error: shouldn't have degraded by more than a few mm given sub-pixel-scale noise.
    CHECK(std::abs(result.p_camera.x - expected_p[0]) < 0.01);
    CHECK(std::abs(result.p_camera.y - expected_p[1]) < 0.01);
    CHECK(std::abs(result.p_camera.z - expected_p[2]) < 0.01);
}

TEST_CASE("solve_board_pnp_from_corners rejects a near-frontal board (tilt gate)", "[phase6][board_pnp]") {
    SyntheticScene scene;
    const cv::Matx33d R_true = rx_for_tilt_deg(5.0);  // below the 15deg default gate
    const cv::Vec3d t_true(0.0, 0.0, 1.3);

    const auto corners = project_synthetic_corners(scene, R_true, t_true);
    const auto result = solve_board_pnp_from_corners(
        corners, scene.nx, scene.ny, scene.square_m, scene.K, scene.x_b, 1.0, 15.0);

    REQUIRE(result.board_found);
    CHECK_FALSE(result.accepted);
    CHECK_THAT(result.pose.tilt_deg, WithinAbs(5.0, 1e-3));
    CHECK(result.reject_reason.find("near-frontal") != std::string::npos);
}

TEST_CASE("solve_board_pnp_from_corners rejects excessive reprojection error", "[phase6][board_pnp]") {
    SyntheticScene scene;
    const cv::Matx33d R_true = rx_for_tilt_deg(40.0);
    const cv::Vec3d t_true(0.0, 0.0, 1.3);

    auto corners = project_synthetic_corners(scene, R_true, t_true);
    corners[0].x += 50.0f;  // one grossly corrupted corner (e.g. a mis-detected point)

    const auto result = solve_board_pnp_from_corners(
        corners, scene.nx, scene.ny, scene.square_m, scene.K, scene.x_b, 1.0, 15.0);

    REQUIRE(result.board_found);
    CHECK_FALSE(result.accepted);
    CHECK(result.pose.reprojection_rms_px > 1.0);
    CHECK(result.reject_reason.find("reprojection RMS") != std::string::npos);
}

TEST_CASE("solve_board_pnp_from_corners rejects a mismatched corner count", "[phase6][board_pnp]") {
    SyntheticScene scene;
    std::vector<cv::Point2f> too_few_corners(10, cv::Point2f(0, 0));  // nx*ny == 35, not 10

    const auto result = solve_board_pnp_from_corners(
        too_few_corners, scene.nx, scene.ny, scene.square_m, scene.K, scene.x_b, 1.0, 15.0);

    CHECK_FALSE(result.accepted);
    CHECK(result.reject_reason.find("Corner count") != std::string::npos);
}
