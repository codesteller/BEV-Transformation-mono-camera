#pragma once

#include <string>

namespace bev {

struct ChessboardConfig {
    int inner_corners_x = 8;
    int inner_corners_y = 6;
    double square_size_m = 0.035;
};

struct FloorMarkerConfig {
    double marker_grid_width_m = 2.0;
    double marker_grid_height_m = 2.0;
    double camera_to_ground_m = 1.5;
};

struct RuntimeConfig {
    int camera_device_index = 0;
    int frame_width = 1920;
    int frame_height = 1080;
    std::string intrinsics_yaml = "config/intrinsics.yaml";
    std::string extrinsics_yaml = "config/extrinsics.yaml";
};

}  // namespace bev
