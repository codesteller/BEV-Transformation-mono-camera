#pragma once

// Output schema for the solved radar<->camera extrinsic (radcam_calibplan.md §3.2/§12 item 5),
// written/read via yaml-cpp. Deliberately kept separate from bev/calibration_io.hpp (the hand-rolled
// reader calibration_tool_qt's Intrinsics/Homography tabs already use) so bev_runner and any other
// consumer of that header don't inherit a yaml-cpp dependency until they actually need this file.

#include "bev/radarcam/types.hpp"

#include <yaml-cpp/yaml.h>

#include <opencv2/core.hpp>

#include <array>
#include <fstream>
#include <string>

namespace bev {
namespace radarcam {

struct ExtrinsicsData {
    cv::Matx33d R = cv::Matx33d::eye();  // radar -> camera rotation
    cv::Vec3d t{0.0, 0.0, 0.0};           // radar -> camera translation, meters
    cv::Vec3d euler_deg{0.0, 0.0, 0.0};   // roll, pitch, yaw, degrees

    int n_inliers = 0;
    int n_total = 0;
    double rms_residual_m = 0.0;
    double rms_reprojection_px = 0.0;

    // §8: estimated vs. tape-measured reflector offset -- a large drift flags a bad caliper measurement.
    cv::Vec3d x_b_measured_m{0.0, 0.0, 0.0};
    cv::Vec3d x_b_estimated_m{0.0, 0.0, 0.0};
    double x_b_drift_m = 0.0;

    std::array<StageFunnelCount, 7> funnel{};  // aggregate across all captures, in §6 stage order
};

inline bool save_extrinsics_yaml(const std::string& path, const ExtrinsicsData& data, std::string& error) {
    YAML::Emitter out;
    out << YAML::BeginMap;

    out << YAML::Key << "R" << YAML::Value << YAML::Flow << YAML::BeginSeq;
    for (int r = 0; r < 3; ++r) {
        for (int c = 0; c < 3; ++c) out << data.R(r, c);
    }
    out << YAML::EndSeq;

    out << YAML::Key << "t" << YAML::Value << YAML::Flow << YAML::BeginSeq << data.t[0] << data.t[1]
        << data.t[2] << YAML::EndSeq;

    out << YAML::Key << "euler_deg" << YAML::Value << YAML::Flow << YAML::BeginSeq << data.euler_deg[0]
        << data.euler_deg[1] << data.euler_deg[2] << YAML::EndSeq;

    out << YAML::Key << "n_inliers" << YAML::Value << data.n_inliers;
    out << YAML::Key << "n_total" << YAML::Value << data.n_total;
    out << YAML::Key << "rms_residual_m" << YAML::Value << data.rms_residual_m;
    out << YAML::Key << "rms_reprojection_px" << YAML::Value << data.rms_reprojection_px;

    out << YAML::Key << "x_b_measured_m" << YAML::Value << YAML::Flow << YAML::BeginSeq
        << data.x_b_measured_m[0] << data.x_b_measured_m[1] << data.x_b_measured_m[2] << YAML::EndSeq;
    out << YAML::Key << "x_b_estimated_m" << YAML::Value << YAML::Flow << YAML::BeginSeq
        << data.x_b_estimated_m[0] << data.x_b_estimated_m[1] << data.x_b_estimated_m[2] << YAML::EndSeq;
    out << YAML::Key << "x_b_drift_m" << YAML::Value << data.x_b_drift_m;

    out << YAML::Key << "funnel" << YAML::Value << YAML::BeginSeq;
    for (const auto& stage : data.funnel) {
        out << YAML::BeginMap;
        out << YAML::Key << "stage" << YAML::Value << stage.stage_name;
        out << YAML::Key << "n_entering" << YAML::Value << stage.n_entering;
        out << YAML::Key << "n_surviving" << YAML::Value << stage.n_surviving;
        out << YAML::EndMap;
    }
    out << YAML::EndSeq;

    out << YAML::EndMap;

    std::ofstream fs(path, std::ios::out | std::ios::trunc);
    if (!fs.is_open()) {
        error = "Could not open file for writing: " + path;
        return false;
    }
    fs << out.c_str();
    return true;
}

inline bool load_extrinsics_yaml(const std::string& path, ExtrinsicsData& out, std::string& error) {
    YAML::Node root;
    try {
        root = YAML::LoadFile(path);
    } catch (const std::exception& e) {
        error = std::string("Failed to load/parse YAML: ") + e.what();
        return false;
    }

    if (!root["R"] || root["R"].size() != 9) {
        error = "R must have 9 values.";
        return false;
    }
    for (int i = 0; i < 9; ++i) out.R(i / 3, i % 3) = root["R"][i].as<double>();

    if (!root["t"] || root["t"].size() != 3) {
        error = "t must have 3 values.";
        return false;
    }
    out.t = cv::Vec3d(root["t"][0].as<double>(), root["t"][1].as<double>(), root["t"][2].as<double>());

    if (root["euler_deg"] && root["euler_deg"].size() == 3) {
        out.euler_deg = cv::Vec3d(root["euler_deg"][0].as<double>(), root["euler_deg"][1].as<double>(),
            root["euler_deg"][2].as<double>());
    }

    if (root["n_inliers"]) out.n_inliers = root["n_inliers"].as<int>();
    if (root["n_total"]) out.n_total = root["n_total"].as<int>();
    if (root["rms_residual_m"]) out.rms_residual_m = root["rms_residual_m"].as<double>();
    if (root["rms_reprojection_px"]) out.rms_reprojection_px = root["rms_reprojection_px"].as<double>();

    if (root["x_b_measured_m"] && root["x_b_measured_m"].size() == 3) {
        out.x_b_measured_m = cv::Vec3d(root["x_b_measured_m"][0].as<double>(),
            root["x_b_measured_m"][1].as<double>(), root["x_b_measured_m"][2].as<double>());
    }
    if (root["x_b_estimated_m"] && root["x_b_estimated_m"].size() == 3) {
        out.x_b_estimated_m = cv::Vec3d(root["x_b_estimated_m"][0].as<double>(),
            root["x_b_estimated_m"][1].as<double>(), root["x_b_estimated_m"][2].as<double>());
    }
    if (root["x_b_drift_m"]) out.x_b_drift_m = root["x_b_drift_m"].as<double>();

    if (root["funnel"] && root["funnel"].IsSequence()) {
        size_t i = 0;
        for (const auto& node : root["funnel"]) {
            if (i >= out.funnel.size()) break;
            out.funnel[i].stage_name = node["stage"] ? node["stage"].as<std::string>() : "";
            out.funnel[i].n_entering = node["n_entering"] ? node["n_entering"].as<int>() : 0;
            out.funnel[i].n_surviving = node["n_surviving"] ? node["n_surviving"].as<int>() : 0;
            ++i;
        }
    }

    return true;
}

}  // namespace radarcam
}  // namespace bev
