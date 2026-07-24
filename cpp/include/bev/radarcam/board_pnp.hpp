#pragma once

// Camera-side pipeline (radcam_calibplan.md §5): checkerboard detection + PnP against a supplied
// camera matrix + reflector position prediction via the caliper-measured board-frame offset X_B.
// Quality-gates a capture per spec: full-board detection, PnP reprojection RMS, and board tilt.
//
// Rectify-before-fusion: `camera_matrix` must be the "effective" matrix for an already-rectified
// (undistorted) image -- projection_matrix if the intrinsics YAML provided one, else the plain
// camera_matrix -- with zero distortion, exactly matching the convention already established in
// HomographyTab::estimate_camera_pose_pnp (cpp/apps/calibration_tool_qt.cpp). Board detection and
// PnP both assume the image passed in has already been undistorted with that same matrix.
//
// PnP method: cv::SOLVEPNP_IPPE (the spec's choice -- markedly more robust than a generic iterative
// solve for a *planar* target, especially at the poorly-conditioned near-frontal poses this module
// explicitly wants to detect/reject) seeded, then polished with cv::solvePnPRefineLM -- keeping the
// repo's existing "iterative refine" habit (HomographyTab uses SOLVEPNP_ITERATIVE) for the final
// answer while fixing IPPE's specific advantage: robust initialization on planar boards.
//
// Split into a corner-level function (solve_board_pnp_from_corners) and an image-level wrapper
// (detect_board_and_solve_pnp) so the PnP/projection/gating math is testable against synthetic
// corners with a known ground-truth pose, with no real image or camera required.

#include <opencv2/calib3d.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <cmath>
#include <string>
#include <vector>

namespace bev {
namespace radarcam {

struct BoardPose {
    cv::Matx33d R = cv::Matx33d::eye();  // board -> camera rotation
    cv::Vec3d t{0.0, 0.0, 0.0};           // board -> camera translation, meters
    double reprojection_rms_px = 0.0;
    double tilt_deg = 0.0;  // deviation from a dead-on-frontal board pose, degrees
};

struct BoardDetectionResult {
    bool board_found = false;
    bool accepted = false;  // false if detection failed or any quality gate rejected the capture
    std::string reject_reason;
    std::vector<cv::Point2f> corners;  // sub-pixel refined image points, if found
    BoardPose pose;
    cv::Point3d p_camera{0.0, 0.0, 0.0};  // predicted reflector position in camera frame: R_k*X_B + t_k
};

// Standard row-major checkerboard object points on the z=0 plane, matching
// cv::findChessboardCorners' corner ordering.
inline std::vector<cv::Point3f> board_object_points(
    int inner_corners_x, int inner_corners_y, double square_size_m) {
    std::vector<cv::Point3f> object_points;
    object_points.reserve(static_cast<size_t>(inner_corners_x) * static_cast<size_t>(inner_corners_y));
    for (int row = 0; row < inner_corners_y; ++row) {
        for (int col = 0; col < inner_corners_x; ++col) {
            object_points.emplace_back(
                static_cast<float>(col * square_size_m), static_cast<float>(row * square_size_m), 0.0f);
        }
    }
    return object_points;
}

// Core PnP + reflector-projection + quality-gate logic, operating on already-known image points
// (sub-pixel corners). No image, camera, or detection involved -- independently testable with
// synthetic corners produced via cv::projectPoints from a known ground-truth pose.
inline BoardDetectionResult solve_board_pnp_from_corners(const std::vector<cv::Point2f>& corners,
    int inner_corners_x, int inner_corners_y, double square_size_m, const cv::Matx33d& camera_matrix,
    const cv::Vec3d& x_b_measured_m, double pnp_reproj_rms_max_px, double board_min_tilt_deg) {
    BoardDetectionResult result;
    result.board_found = true;
    result.corners = corners;

    const auto object_points = board_object_points(inner_corners_x, inner_corners_y, square_size_m);
    if (object_points.size() != corners.size()) {
        result.accepted = false;
        result.reject_reason = "Corner count does not match inner_corners_x * inner_corners_y.";
        return result;
    }

    const cv::Mat camera_matrix_mat(camera_matrix);
    const cv::Mat zero_dist = cv::Mat::zeros(4, 1, CV_64F);

    cv::Mat rvec, tvec;
    const bool ippe_ok = cv::solvePnP(
        object_points, corners, camera_matrix_mat, zero_dist, rvec, tvec, false, cv::SOLVEPNP_IPPE);
    if (!ippe_ok) {
        result.accepted = false;
        result.reject_reason = "solvePnP (IPPE) failed to converge.";
        return result;
    }
    cv::solvePnPRefineLM(object_points, corners, camera_matrix_mat, zero_dist, rvec, tvec);

    cv::Mat R_mat;
    cv::Rodrigues(rvec, R_mat);
    result.pose.R = cv::Matx33d(R_mat);
    result.pose.t = cv::Vec3d(tvec.at<double>(0), tvec.at<double>(1), tvec.at<double>(2));

    std::vector<cv::Point2f> reprojected;
    cv::projectPoints(object_points, rvec, tvec, camera_matrix_mat, zero_dist, reprojected);
    double sq_sum = 0.0;
    for (size_t i = 0; i < corners.size(); ++i) {
        const double dx = reprojected[i].x - corners[i].x;
        const double dy = reprojected[i].y - corners[i].y;
        sq_sum += dx * dx + dy * dy;
    }
    result.pose.reprojection_rms_px = std::sqrt(sq_sum / static_cast<double>(corners.size()));

    // Board frame's local +z points "out of the board face toward the camera" (spec §3.3). A
    // dead-on-frontal board has that axis, rotated into the camera frame, antiparallel to the
    // camera's own +z (viewing) axis -- so tilt is the deviation from that antiparallel alignment.
    const cv::Vec3d normal_cam = result.pose.R * cv::Vec3d(0.0, 0.0, 1.0);
    const double cos_angle = normal_cam.dot(cv::Vec3d(0.0, 0.0, -1.0));
    result.pose.tilt_deg = std::acos(std::clamp(cos_angle, -1.0, 1.0)) * 180.0 / CV_PI;

    const cv::Vec3d p_cam_vec = result.pose.R * x_b_measured_m + result.pose.t;
    result.p_camera = cv::Point3d(p_cam_vec[0], p_cam_vec[1], p_cam_vec[2]);

    if (result.pose.reprojection_rms_px > pnp_reproj_rms_max_px) {
        result.accepted = false;
        result.reject_reason = "PnP reprojection RMS (" + std::to_string(result.pose.reprojection_rms_px) +
            " px) exceeds threshold (" + std::to_string(pnp_reproj_rms_max_px) + " px).";
        return result;
    }
    if (result.pose.tilt_deg < board_min_tilt_deg) {
        result.accepted = false;
        result.reject_reason = "Board viewing angle too near-frontal (" +
            std::to_string(result.pose.tilt_deg) + " deg tilt, minimum " +
            std::to_string(board_min_tilt_deg) + " deg).";
        return result;
    }

    result.accepted = true;
    return result;
}

// Full image-level pipeline: detect the checkerboard, refine to sub-pixel, then delegate to
// solve_board_pnp_from_corners. `image` is assumed already rectified (undistorted).
inline BoardDetectionResult detect_board_and_solve_pnp(const cv::Mat& image, int inner_corners_x,
    int inner_corners_y, double square_size_m, const cv::Matx33d& camera_matrix,
    const cv::Vec3d& x_b_measured_m, double pnp_reproj_rms_max_px, double board_min_tilt_deg) {
    BoardDetectionResult result;

    const cv::Size board_size(inner_corners_x, inner_corners_y);
    std::vector<cv::Point2f> corners;
    const bool found = cv::findChessboardCorners(
        image, board_size, corners, cv::CALIB_CB_ADAPTIVE_THRESH | cv::CALIB_CB_NORMALIZE_IMAGE);

    if (!found) {
        result.board_found = false;
        result.accepted = false;
        result.reject_reason = "Board not fully detected.";
        return result;
    }

    cv::Mat gray;
    if (image.channels() == 3) {
        cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
    } else {
        gray = image;
    }
    cv::cornerSubPix(gray, corners, cv::Size(11, 11), cv::Size(-1, -1),
        cv::TermCriteria(cv::TermCriteria::EPS | cv::TermCriteria::MAX_ITER, 30, 0.1));

    return solve_board_pnp_from_corners(corners, inner_corners_x, inner_corners_y, square_size_m,
        camera_matrix, x_b_measured_m, pnp_reproj_rms_max_px, board_min_tilt_deg);
}

}  // namespace radarcam
}  // namespace bev
