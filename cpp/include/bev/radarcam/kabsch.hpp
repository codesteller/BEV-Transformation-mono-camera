#pragma once

// Closed-form rigid transform solver (Kabsch/Umeyama with determinant correction), radcam_calibplan
// §7.1. Minimizes sum ||p_i - (R*q_i + t)||^2 for index-matched point clouds {p_i} (camera) and
// {q_i} (radar). Globally optimal, no initialization, no iteration. Requires >=3 non-collinear
// correspondences.
//
// The determinant correction (diag(1,1,d)) is mandatory, not optional: V*U^T is orthogonal but may
// be a reflection (det=-1), which is physically meaningless for a rigidly mounted sensor. Absorbing
// the flip into the smallest-singular-value axis -- the direction the data trusts least -- is what
// the spec calls out explicitly. Near-coplanar/near-collinear input is exactly when this matters:
// det(R) must still come out +1.

#include <opencv2/core.hpp>

#include <cstddef>
#include <vector>

namespace bev {
namespace radarcam {

struct RigidTransform {
    cv::Matx33d R = cv::Matx33d::eye();
    cv::Vec3d t{0.0, 0.0, 0.0};
};

// Requires p_camera.size() == q_radar.size() >= 3. Returns false (leaving `out` untouched) otherwise.
inline bool solve_kabsch(
    const std::vector<cv::Point3d>& p_camera, const std::vector<cv::Point3d>& q_radar, RigidTransform& out) {
    if (p_camera.size() != q_radar.size() || p_camera.size() < 3) {
        return false;
    }
    const size_t n = p_camera.size();

    cv::Point3d centroid_p(0.0, 0.0, 0.0), centroid_q(0.0, 0.0, 0.0);
    for (size_t i = 0; i < n; ++i) {
        centroid_p += p_camera[i];
        centroid_q += q_radar[i];
    }
    centroid_p *= (1.0 / static_cast<double>(n));
    centroid_q *= (1.0 / static_cast<double>(n));

    // H = Q^T * P (3x3 cross-covariance): H(i,j) = sum_k Q_k[i] * P_k[j].
    cv::Matx33d H = cv::Matx33d::zeros();
    for (size_t i = 0; i < n; ++i) {
        const cv::Point3d P = p_camera[i] - centroid_p;
        const cv::Point3d Q = q_radar[i] - centroid_q;
        H(0, 0) += Q.x * P.x;
        H(0, 1) += Q.x * P.y;
        H(0, 2) += Q.x * P.z;
        H(1, 0) += Q.y * P.x;
        H(1, 1) += Q.y * P.y;
        H(1, 2) += Q.y * P.z;
        H(2, 0) += Q.z * P.x;
        H(2, 1) += Q.z * P.y;
        H(2, 2) += Q.z * P.z;
    }

    cv::Mat U, S, Vt;
    cv::SVD::compute(cv::Mat(H), S, U, Vt, cv::SVD::FULL_UV);
    const cv::Mat V = Vt.t();

    const double d = (cv::determinant(V * U.t()) < 0.0) ? -1.0 : 1.0;
    cv::Mat D = cv::Mat::eye(3, 3, CV_64F);
    D.at<double>(2, 2) = d;

    const cv::Mat R_mat = V * D * U.t();
    out.R = cv::Matx33d(R_mat);

    const cv::Vec3d cq(centroid_q.x, centroid_q.y, centroid_q.z);
    const cv::Vec3d cp(centroid_p.x, centroid_p.y, centroid_p.z);
    out.t = cp - out.R * cq;

    return true;
}

}  // namespace radarcam
}  // namespace bev
