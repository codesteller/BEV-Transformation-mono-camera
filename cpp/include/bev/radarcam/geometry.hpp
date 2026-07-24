#pragma once

// Radar spherical <-> Cartesian conversion, per radcam_calibplan.md §3.3's fixed convention:
//   radar frame: x forward, y left, z up.
//   azimuth theta measured from +x toward +y; elevation phi from the horizontal plane, positive up.
//   x = r * cos(phi) * cos(theta)
//   y = r * cos(phi) * sin(theta)
//   z = r * sin(phi)
// Every downstream module (clutter filter, solver, Ceres residuals) must go through these
// functions rather than reimplementing the trig inline, so the convention can't silently drift.

#include <opencv2/core.hpp>

#include <algorithm>
#include <cmath>

namespace bev {
namespace radarcam {

struct SphericalCoord {
    double range_m = 0.0;
    double azimuth_rad = 0.0;
    double elevation_rad = 0.0;
};

inline cv::Point3d spherical_to_cartesian(double range_m, double azimuth_rad, double elevation_rad) {
    const double cos_el = std::cos(elevation_rad);
    return cv::Point3d(range_m * cos_el * std::cos(azimuth_rad), range_m * cos_el * std::sin(azimuth_rad),
        range_m * std::sin(elevation_rad));
}

inline cv::Point3d spherical_to_cartesian(const SphericalCoord& s) {
    return spherical_to_cartesian(s.range_m, s.azimuth_rad, s.elevation_rad);
}

inline SphericalCoord cartesian_to_spherical(const cv::Point3d& p) {
    SphericalCoord s;
    s.range_m = std::sqrt(p.x * p.x + p.y * p.y + p.z * p.z);
    s.azimuth_rad = std::atan2(p.y, p.x);
    s.elevation_rad = (s.range_m > 0.0) ? std::asin(std::clamp(p.z / s.range_m, -1.0, 1.0)) : 0.0;
    return s;
}

}  // namespace radarcam
}  // namespace bev
