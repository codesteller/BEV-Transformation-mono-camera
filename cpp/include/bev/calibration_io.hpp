#pragma once

// Shared, Qt-free readers for the YAML files calibration_tool_qt writes, so bev_runner (and any
// other consumer) parses the exact same file format without duplicating/drifting from the tool's
// own understanding of it. Not a general YAML parser -- tailored to the specific hand-written
// schema IntrinsicsTab::calibrate_and_save() and HomographyTab::solve_and_save() produce.

#include <opencv2/core.hpp>

#include <cstdlib>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

namespace bev {

struct IntrinsicsData {
    int image_width = 0;
    int image_height = 0;
    cv::Mat camera_matrix;      // 3x3 CV_64F (K)
    cv::Mat dist_coeffs;        // Nx1 CV_64F (D)
    cv::Mat projection_matrix;  // 3x3 CV_64F, optional (top-left of ROS-style 3x4 P)
};

struct HomographyData {
    int image_width = 0;
    int image_height = 0;
    double plane_width_m = 0.0;
    double plane_height_m = 0.0;
    double camera_height_m = 0.0;
    cv::Point2f camera_ground_xy{};
    cv::Mat homography_matrix;  // 3x3 CV_64F, maps rectified-image pixels -> world meters
    std::string intrinsics_source;
};

// Default home for all calibration assets: ${HOME}/.calibration/openadas/cam<id>/. The cam<id>
// segment keeps multiple cameras' intrinsics/homography/frames from clobbering each other.
inline std::string default_camera_asset_dir(int camera_id) {
    const char* home = std::getenv("HOME");
    const std::string home_str = (home != nullptr) ? home : ".";
    return home_str + "/.calibration/openadas/cam" + std::to_string(camera_id);
}

namespace detail {

// Finds the next "[ ... ]" list starting at or after search_pos and parses its comma-separated
// numbers. Works for both "key: [v0, v1, ...]" and "key:\n  rows: r\n  cols: c\n  data: [...]"
// shapes, since it only cares about locating the bracket pair -- whatever text (including
// "data:") sits between search_pos and the '[' is irrelevant.
inline std::vector<double> extract_bracket_list(const std::string& text, size_t search_pos) {
    std::vector<double> values;
    if (search_pos == std::string::npos) {
        return values;
    }
    const size_t open = text.find('[', search_pos);
    const size_t close = (open == std::string::npos) ? std::string::npos : text.find(']', open);
    if (open == std::string::npos || close == std::string::npos) {
        return values;
    }
    std::stringstream ss(text.substr(open + 1, close - open - 1));
    std::string token;
    while (std::getline(ss, token, ',')) {
        try {
            values.push_back(std::stod(token));
        } catch (const std::exception&) {
        }
    }
    return values;
}

inline int extract_int_after(const std::string& text, const std::string& key) {
    const size_t pos = text.find(key);
    if (pos == std::string::npos) {
        return -1;
    }
    try {
        return std::stoi(text.substr(pos + key.size()));
    } catch (const std::exception&) {
        return -1;
    }
}

inline double extract_double_after(const std::string& text, const std::string& key) {
    const size_t pos = text.find(key);
    if (pos == std::string::npos) {
        return 0.0;
    }
    try {
        return std::stod(text.substr(pos + key.size()));
    } catch (const std::exception&) {
        return 0.0;
    }
}

inline bool read_file(const std::string& path, std::string& out_text, std::string& error) {
    std::ifstream fs(path);
    if (!fs.is_open()) {
        error = "Could not open file.";
        return false;
    }
    std::ostringstream buffer;
    buffer << fs.rdbuf();
    out_text = buffer.str();
    return true;
}

}  // namespace detail

inline bool load_intrinsics_yaml(const std::string& path, IntrinsicsData& out, std::string& error) {
    std::string text;
    if (!detail::read_file(path, text, error)) {
        return false;
    }

    out.image_width = detail::extract_int_after(text, "image_width:");
    out.image_height = detail::extract_int_after(text, "image_height:");
    if (out.image_width <= 0 || out.image_height <= 0) {
        error = "Missing or invalid image_width/image_height.";
        return false;
    }

    const auto k_values = detail::extract_bracket_list(text, text.find("camera_matrix:"));
    if (k_values.size() != 9) {
        error = "camera_matrix must have 9 values.";
        return false;
    }
    out.camera_matrix = cv::Mat(3, 3, CV_64F);
    for (int i = 0; i < 9; ++i) {
        out.camera_matrix.at<double>(i / 3, i % 3) = k_values[static_cast<size_t>(i)];
    }

    const auto d_values = detail::extract_bracket_list(text, text.find("distortion_coefficients:"));
    if (d_values.empty()) {
        error = "distortion_coefficients missing.";
        return false;
    }
    out.dist_coeffs = cv::Mat(static_cast<int>(d_values.size()), 1, CV_64F);
    for (size_t i = 0; i < d_values.size(); ++i) {
        out.dist_coeffs.at<double>(static_cast<int>(i), 0) = d_values[i];
    }

    const auto p_values = detail::extract_bracket_list(text, text.find("projection_matrix:"));
    if (p_values.size() == 12) {
        out.projection_matrix = cv::Mat(3, 3, CV_64F);
        for (int r = 0; r < 3; ++r) {
            for (int c = 0; c < 3; ++c) {
                out.projection_matrix.at<double>(r, c) = p_values[static_cast<size_t>(r * 4 + c)];
            }
        }
    }

    return true;
}

inline bool load_homography_yaml(const std::string& path, HomographyData& out, std::string& error) {
    std::string text;
    if (!detail::read_file(path, text, error)) {
        return false;
    }

    out.image_width = detail::extract_int_after(text, "image_width:");
    out.image_height = detail::extract_int_after(text, "image_height:");
    if (out.image_width <= 0 || out.image_height <= 0) {
        error = "Missing or invalid image_width/image_height.";
        return false;
    }

    out.plane_width_m = detail::extract_double_after(text, "plane_width_m:");
    out.plane_height_m = detail::extract_double_after(text, "plane_height_m:");
    if (out.plane_width_m <= 0.0 || out.plane_height_m <= 0.0) {
        error = "Missing or invalid plane_width_m/plane_height_m.";
        return false;
    }

    out.camera_height_m = detail::extract_double_after(text, "camera_to_ground_m:");

    const auto xy = detail::extract_bracket_list(text, text.find("camera_ground_xy_m:"));
    if (xy.size() == 2) {
        out.camera_ground_xy = cv::Point2f(static_cast<float>(xy[0]), static_cast<float>(xy[1]));
    }

    const auto h_values = detail::extract_bracket_list(text, text.find("homography_matrix:"));
    if (h_values.size() != 9) {
        error = "homography_matrix must have 9 values.";
        return false;
    }
    out.homography_matrix = cv::Mat(3, 3, CV_64F);
    for (int i = 0; i < 9; ++i) {
        out.homography_matrix.at<double>(i / 3, i % 3) = h_values[static_cast<size_t>(i)];
    }

    const size_t src_pos = text.find("intrinsics_source:");
    if (src_pos != std::string::npos) {
        const size_t q1 = text.find('"', src_pos);
        const size_t q2 = (q1 == std::string::npos) ? std::string::npos : text.find('"', q1 + 1);
        if (q1 != std::string::npos && q2 != std::string::npos) {
            out.intrinsics_source = text.substr(q1 + 1, q2 - q1 - 1);
        }
    }

    return true;
}

}  // namespace bev
