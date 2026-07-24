#pragma once

// Nonlinear refinement (radcam_calibplan §8), seeded from the closed-form Kabsch/RANSAC result.
//
// Mode A -- refine (R, t) only. Rotation is parametrized as a Rodrigues/angle-axis vector (3
// params) so the SO(3) constraint is structural, never enforced after the fact. Each residual is
// p_camera_k - (R*q_radar_k + t), wrapped in a Huber loss. Optional per-axis weighting (radar is
// excellent in range/mediocre in angle; camera PnP is excellent in angle/weak in depth) is a fixed
// diagonal scale applied inside the functor -- not a separate cost function.
//
// Mode B -- jointly estimates the 9th/10th/11th parameters: the reflector-offset X_B (3 params),
// shared across every correspondence's residual block. This exists because an error in the
// measured reflector offset is the one systematic error the residuals otherwise can't detect (see
// board_pose comment below). Because X_B enters through R_k (the *board* pose at capture k, fixed
// per correspondence -- not the R being solved for), the residual must recompute R_k*X_B + t_k
// inside the functor rather than reuse Mode A's pre-collapsed p_camera_k. Ceres's cost-function
// arity is fixed at compile time (2 parameter blocks of size {3,3} for Mode A; 3 blocks of size
// {3,3,3} for Mode B), so these are necessarily two distinct functor templates sharing the same
// underlying residual math, not one functor with an optional block.
//
// Per spec: "run Mode B as a diagnostic regardless" of which mode's result is treated as primary --
// the caller decides what to do with x_b_estimated vs. the tape-measured x_b_measured (report the
// drift; see bev::radarcam::ExtrinsicsData).

#include "bev/radarcam/board_pnp.hpp"
#include "bev/radarcam/kabsch.hpp"

#include <ceres/ceres.h>
#include <ceres/rotation.h>

#include <opencv2/calib3d.hpp>
#include <opencv2/core.hpp>

#include <string>
#include <vector>

namespace bev {
namespace radarcam {

// One correspondence's fixed inputs to refinement. `board_pose` (the board's pose AT THIS CAPTURE,
// R_k/t_k) is retained specifically for Mode B -- Mode A never touches it, since it only needs the
// already-collapsed p_camera (== R_k*X_B_measured + t_k from the camera-side pipeline).
struct RefinementInput {
    cv::Point3d p_camera{0.0, 0.0, 0.0};
    cv::Point3d q_radar{0.0, 0.0, 0.0};
    BoardPose board_pose;
};

struct RefinementResult {
    bool success = false;
    RigidTransform transform;
    cv::Vec3d x_b_estimated{0.0, 0.0, 0.0};  // only populated/meaningful by refine_mode_b
    double final_cost = 0.0;
    std::string brief_report;
};

namespace detail {

struct ModeAResidual {
    ModeAResidual(const cv::Point3d& p_camera, const cv::Point3d& q_radar, const cv::Vec3d& weight)
        : p_camera_(p_camera), q_radar_(q_radar), weight_(weight) {}

    template <typename T>
    bool operator()(const T* const angle_axis, const T* const t, T* residual) const {
        T q[3] = {T(q_radar_.x), T(q_radar_.y), T(q_radar_.z)};
        T q_rot[3];
        ceres::AngleAxisRotatePoint(angle_axis, q, q_rot);
        residual[0] = T(weight_[0]) * (T(p_camera_.x) - (q_rot[0] + t[0]));
        residual[1] = T(weight_[1]) * (T(p_camera_.y) - (q_rot[1] + t[1]));
        residual[2] = T(weight_[2]) * (T(p_camera_.z) - (q_rot[2] + t[2]));
        return true;
    }

    static ceres::CostFunction* Create(
        const cv::Point3d& p_camera, const cv::Point3d& q_radar, const cv::Vec3d& weight) {
        return new ceres::AutoDiffCostFunction<ModeAResidual, 3, 3, 3>(
            new ModeAResidual(p_camera, q_radar, weight));
    }

    cv::Point3d p_camera_;
    cv::Point3d q_radar_;
    cv::Vec3d weight_;
};

struct ModeBResidual {
    ModeBResidual(const cv::Matx33d& R_k, const cv::Vec3d& t_k, const cv::Point3d& q_radar, const cv::Vec3d& weight)
        : R_k_(R_k), t_k_(t_k), q_radar_(q_radar), weight_(weight) {}

    template <typename T>
    bool operator()(const T* const angle_axis, const T* const t, const T* const x_b, T* residual) const {
        // Camera-side prediction recomputed here (not reused from p_camera) because it depends on
        // the X_B parameter being solved for: R_k * X_B + t_k, with R_k/t_k fixed constants.
        T p_pred[3];
        p_pred[0] = T(R_k_(0, 0)) * x_b[0] + T(R_k_(0, 1)) * x_b[1] + T(R_k_(0, 2)) * x_b[2] + T(t_k_[0]);
        p_pred[1] = T(R_k_(1, 0)) * x_b[0] + T(R_k_(1, 1)) * x_b[1] + T(R_k_(1, 2)) * x_b[2] + T(t_k_[1]);
        p_pred[2] = T(R_k_(2, 0)) * x_b[0] + T(R_k_(2, 1)) * x_b[1] + T(R_k_(2, 2)) * x_b[2] + T(t_k_[2]);

        T q[3] = {T(q_radar_.x), T(q_radar_.y), T(q_radar_.z)};
        T q_rot[3];
        ceres::AngleAxisRotatePoint(angle_axis, q, q_rot);

        residual[0] = T(weight_[0]) * (p_pred[0] - (q_rot[0] + t[0]));
        residual[1] = T(weight_[1]) * (p_pred[1] - (q_rot[1] + t[1]));
        residual[2] = T(weight_[2]) * (p_pred[2] - (q_rot[2] + t[2]));
        return true;
    }

    static ceres::CostFunction* Create(const cv::Matx33d& R_k, const cv::Vec3d& t_k,
        const cv::Point3d& q_radar, const cv::Vec3d& weight) {
        return new ceres::AutoDiffCostFunction<ModeBResidual, 3, 3, 3, 3>(
            new ModeBResidual(R_k, t_k, q_radar, weight));
    }

    cv::Matx33d R_k_;
    cv::Vec3d t_k_;
    cv::Point3d q_radar_;
    cv::Vec3d weight_;
};

inline void rotation_matrix_to_angle_axis(const cv::Matx33d& R, double angle_axis[3]) {
    cv::Mat rvec;
    cv::Rodrigues(cv::Mat(R), rvec);
    angle_axis[0] = rvec.at<double>(0);
    angle_axis[1] = rvec.at<double>(1);
    angle_axis[2] = rvec.at<double>(2);
}

inline cv::Matx33d angle_axis_to_rotation_matrix(const double angle_axis[3]) {
    cv::Mat rvec = (cv::Mat_<double>(3, 1) << angle_axis[0], angle_axis[1], angle_axis[2]);
    cv::Mat R_mat;
    cv::Rodrigues(rvec, R_mat);
    return cv::Matx33d(R_mat);
}

}  // namespace detail

inline RefinementResult refine_mode_a(const std::vector<RefinementInput>& correspondences,
    const RigidTransform& initial, double huber_delta_m, const cv::Vec3d& weight = cv::Vec3d(1.0, 1.0, 1.0)) {
    RefinementResult result;
    if (correspondences.size() < 3) return result;

    double angle_axis[3];
    detail::rotation_matrix_to_angle_axis(initial.R, angle_axis);
    double t[3] = {initial.t[0], initial.t[1], initial.t[2]};

    ceres::Problem problem;
    for (const auto& c : correspondences) {
        ceres::CostFunction* cost = detail::ModeAResidual::Create(c.p_camera, c.q_radar, weight);
        problem.AddResidualBlock(cost, new ceres::HuberLoss(huber_delta_m), angle_axis, t);
    }

    ceres::Solver::Options options;
    options.linear_solver_type = ceres::DENSE_QR;
    options.minimizer_progress_to_stdout = false;
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);

    result.success = summary.IsSolutionUsable();
    result.transform.R = detail::angle_axis_to_rotation_matrix(angle_axis);
    result.transform.t = cv::Vec3d(t[0], t[1], t[2]);
    result.final_cost = summary.final_cost;
    result.brief_report = summary.BriefReport();
    return result;
}

inline RefinementResult refine_mode_b(const std::vector<RefinementInput>& correspondences,
    const RigidTransform& initial, const cv::Vec3d& x_b_initial, double huber_delta_m,
    const cv::Vec3d& weight = cv::Vec3d(1.0, 1.0, 1.0)) {
    RefinementResult result;
    if (correspondences.size() < 3) return result;

    double angle_axis[3];
    detail::rotation_matrix_to_angle_axis(initial.R, angle_axis);
    double t[3] = {initial.t[0], initial.t[1], initial.t[2]};
    double x_b[3] = {x_b_initial[0], x_b_initial[1], x_b_initial[2]};

    ceres::Problem problem;
    for (const auto& c : correspondences) {
        ceres::CostFunction* cost =
            detail::ModeBResidual::Create(c.board_pose.R, c.board_pose.t, c.q_radar, weight);
        problem.AddResidualBlock(cost, new ceres::HuberLoss(huber_delta_m), angle_axis, t, x_b);
    }

    ceres::Solver::Options options;
    options.linear_solver_type = ceres::DENSE_QR;
    options.minimizer_progress_to_stdout = false;
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);

    result.success = summary.IsSolutionUsable();
    result.transform.R = detail::angle_axis_to_rotation_matrix(angle_axis);
    result.transform.t = cv::Vec3d(t[0], t[1], t[2]);
    result.x_b_estimated = cv::Vec3d(x_b[0], x_b[1], x_b[2]);
    result.final_cost = summary.final_cost;
    result.brief_report = summary.BriefReport();
    return result;
}

}  // namespace radarcam
}  // namespace bev
