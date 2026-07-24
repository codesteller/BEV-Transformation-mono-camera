#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include "bev/radarcam/geometry.hpp"

using Catch::Matchers::WithinAbs;
using bev::radarcam::cartesian_to_spherical;
using bev::radarcam::spherical_to_cartesian;

namespace {
constexpr double kPi = 3.14159265358979323846;
constexpr double kTol = 1e-9;
}  // namespace

TEST_CASE("spherical_to_cartesian matches the spec's fixed convention on axis-aligned cases",
    "[phase2][geometry]") {
    SECTION("straight ahead: az=0, el=0") {
        const auto p = spherical_to_cartesian(10.0, 0.0, 0.0);
        CHECK_THAT(p.x, WithinAbs(10.0, kTol));
        CHECK_THAT(p.y, WithinAbs(0.0, kTol));
        CHECK_THAT(p.z, WithinAbs(0.0, kTol));
    }

    SECTION("90 deg left: az=+90deg, el=0 -> +y (left)") {
        const auto p = spherical_to_cartesian(10.0, kPi / 2.0, 0.0);
        CHECK_THAT(p.x, WithinAbs(0.0, kTol));
        CHECK_THAT(p.y, WithinAbs(10.0, kTol));
        CHECK_THAT(p.z, WithinAbs(0.0, kTol));
    }

    SECTION("directly behind: az=180deg, el=0 -> -x") {
        const auto p = spherical_to_cartesian(10.0, kPi, 0.0);
        CHECK_THAT(p.x, WithinAbs(-10.0, kTol));
        CHECK_THAT(p.y, WithinAbs(0.0, 1e-8));
        CHECK_THAT(p.z, WithinAbs(0.0, kTol));
    }

    SECTION("straight up: az=0, el=+90deg -> +z") {
        const auto p = spherical_to_cartesian(10.0, 0.0, kPi / 2.0);
        CHECK_THAT(p.x, WithinAbs(0.0, 1e-8));
        CHECK_THAT(p.y, WithinAbs(0.0, kTol));
        CHECK_THAT(p.z, WithinAbs(10.0, kTol));
    }

    SECTION("straight down: az=0, el=-90deg -> -z") {
        const auto p = spherical_to_cartesian(10.0, 0.0, -kPi / 2.0);
        CHECK_THAT(p.x, WithinAbs(0.0, 1e-8));
        CHECK_THAT(p.y, WithinAbs(0.0, kTol));
        CHECK_THAT(p.z, WithinAbs(-10.0, kTol));
    }
}

TEST_CASE("cartesian_to_spherical round-trips over a grid of angles", "[phase2][geometry]") {
    const double range_m = 12.5;
    for (int az_deg = -170; az_deg <= 170; az_deg += 10) {
        for (int el_deg = -80; el_deg <= 80; el_deg += 10) {
            const double az_rad = az_deg * kPi / 180.0;
            const double el_rad = el_deg * kPi / 180.0;

            const auto cart = spherical_to_cartesian(range_m, az_rad, el_rad);
            const auto back = cartesian_to_spherical(cart);

            CAPTURE(az_deg, el_deg);
            CHECK_THAT(back.range_m, WithinAbs(range_m, 1e-9));
            CHECK_THAT(back.azimuth_rad, WithinAbs(az_rad, 1e-9));
            CHECK_THAT(back.elevation_rad, WithinAbs(el_rad, 1e-9));
        }
    }
}

TEST_CASE("cartesian_to_spherical at the origin returns zero range without dividing by zero",
    "[phase2][geometry]") {
    const auto s = cartesian_to_spherical(cv::Point3d(0.0, 0.0, 0.0));
    CHECK_THAT(s.range_m, WithinAbs(0.0, kTol));
    CHECK_THAT(s.elevation_rad, WithinAbs(0.0, kTol));
}
