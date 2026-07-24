#include <catch2/catch_test_macros.hpp>

#include "bev/radarcam/diagnostics_render.hpp"

#include <opencv2/imgcodecs.hpp>

using bev::radarcam::ClutterFunnel;
using bev::radarcam::IterationRecord;
using bev::radarcam::StageFunnelCount;

namespace {
std::vector<cv::Point3d> sample_cloud() {
    return {{5, 0, 0}, {5, 2, 0.2}, {5, -2, 1}, {10, 3, -1}, {10, -3, 2}, {15, 1, 1.5}, {15, -1, -1.5}, {20, 0, 2},
        {8, 4, 0.5}, {8, -4, -0.5}, {12, 0, 3}, {12, 0, -3}};
}
}  // namespace

TEST_CASE("render_reflector_cloud produces a non-empty canvas of the requested size", "[phase11][diagnostics_render]") {
    const auto positions = sample_cloud();
    std::vector<bool> is_inlier(positions.size(), true);
    is_inlier[2] = false;
    is_inlier[7] = false;

    const cv::Mat canvas = bev::radarcam::render_reflector_cloud(positions, is_inlier, 800, 400);
    REQUIRE_FALSE(canvas.empty());
    CHECK(canvas.cols == 800);
    CHECK(canvas.rows == 400);
    CHECK(canvas.type() == CV_8UC3);

    cv::imwrite(std::string(PREVIEW_DIR) + "/render_preview_reflector_cloud.png", canvas);
}

TEST_CASE("render_scatter produces a non-empty canvas and handles empty input", "[phase11][diagnostics_render]") {
    std::vector<double> ranges = {5, 8, 10, 12, 15, 20};
    std::vector<double> residuals = {0.01, 0.02, 0.015, 0.03, 0.025, 0.04};

    const cv::Mat canvas = bev::radarcam::render_scatter(ranges, residuals, "range (m)", "residual (m)");
    REQUIRE_FALSE(canvas.empty());
    CHECK(canvas.cols == 640);
    CHECK(canvas.rows == 400);

    cv::imwrite(std::string(PREVIEW_DIR) + "/render_preview_residual_vs_range.png", canvas);

    const cv::Mat empty_canvas = bev::radarcam::render_scatter({}, {}, "x", "y");
    CHECK_FALSE(empty_canvas.empty());  // still a valid blank canvas, not a crash
}

TEST_CASE("render_reprojection_overlay draws onto a blank canvas when no base image is given",
    "[phase11][diagnostics_render]") {
    const cv::Mat canvas = bev::radarcam::render_reprojection_overlay(cv::Mat(), cv::Point2d(100, 100), cv::Point2d(108, 95));
    REQUIRE_FALSE(canvas.empty());
    CHECK(canvas.type() == CV_8UC3);

    cv::imwrite(std::string(PREVIEW_DIR) + "/render_preview_reprojection_overlay.png", canvas);
}

TEST_CASE("render_inlier_vs_iteration produces a non-empty canvas", "[phase11][diagnostics_render]") {
    std::vector<IterationRecord> iterations;
    IterationRecord r0;
    r0.iteration = 0;
    r0.n_inliers = 30;
    iterations.push_back(r0);
    IterationRecord r1;
    r1.iteration = 1;
    r1.n_inliers = 27;
    iterations.push_back(r1);
    IterationRecord r2;
    r2.iteration = 2;
    r2.n_inliers = 8;
    r2.guardrail_triggered = true;
    iterations.push_back(r2);

    const cv::Mat canvas = bev::radarcam::render_inlier_vs_iteration(iterations);
    REQUIRE_FALSE(canvas.empty());

    cv::imwrite(std::string(PREVIEW_DIR) + "/render_preview_inlier_vs_iteration.png", canvas);
}

TEST_CASE("render_funnel_table produces a non-empty canvas", "[phase11][diagnostics_render]") {
    ClutterFunnel funnel = {StageFunnelCount{"background_subtraction", 100, 95}, StageFunnelCount{"spatial_gate", 95, 60},
        StageFunnelCount{"doppler_gate", 60, 55}, StageFunnelCount{"rcs_gate", 55, 40},
        StageFunnelCount{"ground_plausibility", 40, 38}, StageFunnelCount{"best_in_gate", 38, 30},
        StageFunnelCount{"persistence_aggregation", 30, 27}};

    const cv::Mat canvas = bev::radarcam::render_funnel_table(funnel);
    REQUIRE_FALSE(canvas.empty());

    cv::imwrite(std::string(PREVIEW_DIR) + "/render_preview_funnel_table.png", canvas);
}
