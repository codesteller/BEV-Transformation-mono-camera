#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include "bev/radarcam/kabsch.hpp"
#include "bev/radarcam/ransac.hpp"

#include <opencv2/calib3d.hpp>

#include <algorithm>

using Catch::Matchers::WithinAbs;
using bev::radarcam::RigidTransform;
using bev::radarcam::solve_kabsch;
using bev::radarcam::solve_ransac;

namespace {

// A well-conditioned synthetic point cloud in "radar frame" -- real spread across all three axes
// (range, azimuth-ish lateral offset, height), neither coplanar nor collinear.
std::vector<cv::Point3d> well_conditioned_cloud() {
    return {{5, 0, 0}, {5, 2, 0}, {5, -2, 1}, {10, 3, -1}, {10, -3, 2}, {15, 1, 1.5}, {15, -1, -1.5},
        {20, 0, 2}, {8, 4, 0.5}, {8, -4, -0.5}, {12, 0, 3}, {12, 0, -3}};
}

cv::Point3d apply(const cv::Matx33d& R, const cv::Vec3d& t, const cv::Point3d& q) {
    const cv::Vec3d qv(q.x, q.y, q.z);
    const cv::Vec3d p = R * qv + t;
    return cv::Point3d(p[0], p[1], p[2]);
}

cv::Matx33d rodrigues(double x, double y, double z) {
    cv::Mat rvec = (cv::Mat_<double>(3, 1) << x, y, z);
    cv::Mat R;
    cv::Rodrigues(rvec, R);
    return cv::Matx33d(R);
}
}  // namespace

TEST_CASE("solve_kabsch exactly recovers a known rigid transform (noiseless)", "[phase8][kabsch]") {
    const cv::Matx33d R_true = rodrigues(0.3, -0.2, 0.1);
    const cv::Vec3d t_true(0.5, -0.3, 1.2);

    const auto q_radar = well_conditioned_cloud();
    std::vector<cv::Point3d> p_camera;
    for (const auto& q : q_radar) p_camera.push_back(apply(R_true, t_true, q));

    RigidTransform out;
    REQUIRE(solve_kabsch(p_camera, q_radar, out));

    for (int r = 0; r < 3; ++r) {
        for (int c = 0; c < 3; ++c) {
            CHECK_THAT(out.R(r, c), WithinAbs(R_true(r, c), 1e-9));
        }
    }
    for (int i = 0; i < 3; ++i) {
        CHECK_THAT(out.t[i], WithinAbs(t_true[i], 1e-9));
    }

    // det(R) must be exactly +1, never a reflection.
    CHECK_THAT(cv::determinant(cv::Mat(out.R)), WithinAbs(1.0, 1e-9));
}

TEST_CASE("solve_kabsch degrades gracefully under noise", "[phase8][kabsch]") {
    const cv::Matx33d R_true = rodrigues(0.2, 0.15, -0.1);
    const cv::Vec3d t_true(0.1, 0.2, 0.9);

    const auto q_radar = well_conditioned_cloud();
    std::vector<cv::Point3d> p_camera;
    // Deterministic pseudo-noise (not random -- reproducible), alternating sign, sub-cm scale.
    for (size_t i = 0; i < q_radar.size(); ++i) {
        auto p = apply(R_true, t_true, q_radar[i]);
        const double sign = (i % 2 == 0) ? 1.0 : -1.0;
        p.x += sign * 0.005;
        p.y += -sign * 0.004;
        p.z += sign * 0.003;
        p_camera.push_back(p);
    }

    RigidTransform out;
    REQUIRE(solve_kabsch(p_camera, q_radar, out));

    for (int r = 0; r < 3; ++r) {
        for (int c = 0; c < 3; ++c) {
            CHECK(std::abs(out.R(r, c) - R_true(r, c)) < 0.01);
        }
    }
    for (int i = 0; i < 3; ++i) {
        CHECK(std::abs(out.t[i] - t_true[i]) < 0.01);
    }
}

TEST_CASE("solve_kabsch keeps det(R) = +1 on near-coplanar input", "[phase8][kabsch]") {
    const cv::Matx33d R_true = rodrigues(0.1, -0.05, 0.2);
    const cv::Vec3d t_true(0.0, 0.0, 1.0);

    // Nearly coplanar in radar frame: z is a tiny perturbation around 0, not a real third dimension.
    std::vector<cv::Point3d> q_radar = {{5, 0, 0.001}, {5, 2, -0.001}, {5, -2, 0.0015}, {10, 3, -0.0012},
        {10, -3, 0.0008}, {8, 4, 0.0005}, {8, -4, -0.0007}};
    std::vector<cv::Point3d> p_camera;
    for (const auto& q : q_radar) p_camera.push_back(apply(R_true, t_true, q));

    RigidTransform out;
    REQUIRE(solve_kabsch(p_camera, q_radar, out));

    CHECK_THAT(cv::determinant(cv::Mat(out.R)), WithinAbs(1.0, 1e-6));
}

TEST_CASE("solve_kabsch rejects fewer than 3 correspondences", "[phase8][kabsch]") {
    std::vector<cv::Point3d> p = {{0, 0, 0}, {1, 0, 0}};
    std::vector<cv::Point3d> q = {{0, 0, 0}, {1, 0, 0}};
    RigidTransform out;
    CHECK_FALSE(solve_kabsch(p, q, out));
}

TEST_CASE("solve_ransac rejects injected outliers and recovers the true transform", "[phase8][ransac]") {
    const cv::Matx33d R_true = rodrigues(0.25, -0.1, 0.05);
    const cv::Vec3d t_true(0.2, -0.1, 1.0);

    const auto q_radar_clean = well_conditioned_cloud();
    std::vector<cv::Point3d> p_camera, q_radar;
    for (const auto& q : q_radar_clean) {
        p_camera.push_back(apply(R_true, t_true, q));
        q_radar.push_back(q);
    }

    // Inject 3 wildly wrong correspondences (as if clutter survived §6's filter) among the 12 good ones.
    const std::vector<int> outlier_indices = {2, 5, 9};
    for (int idx : outlier_indices) {
        q_radar[idx] = q_radar[idx] + cv::Point3d(5.0, -4.0, 3.0);
    }

    const auto result = solve_ransac(p_camera, q_radar, /*inlier_threshold_m=*/0.05);
    REQUIRE(result.success);

    // None of the injected outliers should appear in the inlier set.
    for (int idx : outlier_indices) {
        CHECK(std::find(result.inlier_indices.begin(), result.inlier_indices.end(), idx) ==
            result.inlier_indices.end());
    }
    CHECK(result.inlier_indices.size() == q_radar_clean.size() - outlier_indices.size());

    for (int r = 0; r < 3; ++r) {
        for (int c = 0; c < 3; ++c) {
            CHECK_THAT(result.transform.R(r, c), WithinAbs(R_true(r, c), 1e-6));
        }
    }
    for (int i = 0; i < 3; ++i) {
        CHECK_THAT(result.transform.t[i], WithinAbs(t_true[i], 1e-6));
    }
}

TEST_CASE("solve_ransac fails cleanly with fewer than 3 correspondences", "[phase8][ransac]") {
    std::vector<cv::Point3d> p = {{0, 0, 0}, {1, 0, 0}};
    std::vector<cv::Point3d> q = {{0, 0, 0}, {1, 0, 0}};
    const auto result = solve_ransac(p, q, 0.1);
    CHECK_FALSE(result.success);
}
