#pragma once

// RANSAC wrapper around solve_kabsch (radcam_calibplan §7.2). At calibration scale (spec's protocol:
// 30-50 captures), exhaustive enumeration of every 3-point subset is completely tractable
// (C(50,3) = 19600) and strictly better than random sampling: it guarantees finding the true
// best-consensus triple rather than risking an unlucky draw, with no iteration-count/probability
// tuning needed. Would need revisiting (random sampling with a cap) only if correspondence counts
// grew far beyond what this tool's capture protocol ever produces.
//
// Spec's own warning applies regardless of triple-selection strategy: RANSAC is the *last* line of
// defense, cleaning up a handful of bad pairs among many good ones -- it is not a substitute for
// §6's clutter filtering upstream.

#include "bev/radarcam/kabsch.hpp"

#include <opencv2/core.hpp>

#include <cstddef>
#include <vector>

namespace bev {
namespace radarcam {

struct RansacResult {
    bool success = false;
    RigidTransform transform;
    std::vector<int> inlier_indices;  // indices into the input correspondence arrays
};

inline RansacResult solve_ransac(
    const std::vector<cv::Point3d>& p_camera, const std::vector<cv::Point3d>& q_radar, double inlier_threshold_m) {
    RansacResult result;
    const size_t n = p_camera.size();
    if (n != q_radar.size() || n < 3) {
        return result;
    }

    std::vector<int> best_inliers;
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = i + 1; j < n; ++j) {
            for (size_t k = j + 1; k < n; ++k) {
                const std::vector<cv::Point3d> sample_p = {p_camera[i], p_camera[j], p_camera[k]};
                const std::vector<cv::Point3d> sample_q = {q_radar[i], q_radar[j], q_radar[k]};
                RigidTransform candidate;
                if (!solve_kabsch(sample_p, sample_q, candidate)) continue;

                std::vector<int> inliers;
                inliers.reserve(n);
                for (size_t m = 0; m < n; ++m) {
                    const cv::Vec3d qv(q_radar[m].x, q_radar[m].y, q_radar[m].z);
                    const cv::Vec3d predicted = candidate.R * qv + candidate.t;
                    const cv::Point3d predicted_pt(predicted[0], predicted[1], predicted[2]);
                    if (cv::norm(predicted_pt - p_camera[m]) <= inlier_threshold_m) {
                        inliers.push_back(static_cast<int>(m));
                    }
                }
                if (inliers.size() > best_inliers.size()) {
                    best_inliers = std::move(inliers);
                }
            }
        }
    }

    if (best_inliers.size() < 3) {
        return result;
    }

    std::vector<cv::Point3d> inlier_p, inlier_q;
    inlier_p.reserve(best_inliers.size());
    inlier_q.reserve(best_inliers.size());
    for (int idx : best_inliers) {
        inlier_p.push_back(p_camera[idx]);
        inlier_q.push_back(q_radar[idx]);
    }

    RigidTransform refit;
    if (!solve_kabsch(inlier_p, inlier_q, refit)) {
        return result;
    }

    result.success = true;
    result.transform = refit;
    result.inlier_indices = std::move(best_inliers);
    return result;
}

}  // namespace radarcam
}  // namespace bev
