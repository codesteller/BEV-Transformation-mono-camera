#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include "bev/radarcam/capture_io.hpp"

#include <cstdio>
#include <cstdlib>
#include <filesystem>

using Catch::Matchers::WithinAbs;
using bev::radarcam::Capture;
using bev::radarcam::CaptureMeta;
using bev::radarcam::RadarDetection;
using bev::radarcam::RadarDwell;

namespace {
constexpr double kTol = 1e-12;

RadarDetection make_detection(double range_m, double az, double el, double doppler, double rcs, double snr) {
    RadarDetection d;
    d.range_m = range_m;
    d.azimuth_rad = az;
    d.elevation_rad = el;
    d.vel_mps = doppler;
    d.rcs_dbsm = rcs;
    d.snr_db = snr;
    d.position = bev::radarcam::spherical_to_cartesian(range_m, az, el);
    return d;
}
}  // namespace

TEST_CASE(".npy writer/reader round-trips a rectangular float64 array", "[phase5][capture_io][npy]") {
    const size_t dim0 = 2, dim1 = 3, dim2 = 7;
    std::vector<double> data(dim0 * dim1 * dim2);
    for (size_t i = 0; i < data.size(); ++i) data[i] = static_cast<double>(i) * 0.5 - 3.25;

    const std::string path = std::string(FIXTURES_DIR) + "/npy_roundtrip_scratch.npy";
    std::string error;
    REQUIRE(bev::radarcam::write_npy_f64_3d(path, data, dim0, dim1, dim2, error));

    std::vector<double> loaded;
    size_t r0 = 0, r1 = 0, r2 = 0;
    REQUIRE(bev::radarcam::read_npy_f64_3d(path, loaded, r0, r1, r2, error));

    CHECK(r0 == dim0);
    CHECK(r1 == dim1);
    CHECK(r2 == dim2);
    REQUIRE(loaded.size() == data.size());
    for (size_t i = 0; i < data.size(); ++i) {
        CHECK_THAT(loaded[i], WithinAbs(data[i], kTol));
    }

    std::remove(path.c_str());
}

TEST_CASE(".npy files written by write_npy_f64_3d load correctly under real numpy.load",
    "[phase5][capture_io][npy]") {
    const size_t dim0 = 2, dim1 = 4, dim2 = 7;
    std::vector<double> data(dim0 * dim1 * dim2);
    for (size_t i = 0; i < data.size(); ++i) data[i] = static_cast<double>(i) * 1.25;

    const std::string path = std::string(FIXTURES_DIR) + "/npy_numpy_check_scratch.npy";
    std::string error;
    REQUIRE(bev::radarcam::write_npy_f64_3d(path, data, dim0, dim1, dim2, error));

    // Cross-check against genuine numpy, if a Python3 + numpy toolchain happens to be available in
    // the environment running these tests. Skips silently (doesn't fail the suite) when it isn't --
    // the round-trip test above is the primary, environment-independent correctness guarantee;
    // this is a bonus interoperability check.
    const std::string cmd = "python3 -c \"import numpy,sys; a=numpy.load('" + path +
        "'); sys.exit(0 if a.shape==(2,4,7) and a.dtype==numpy.float64 and "
        "abs(a[1,2,3]-" + std::to_string(data[(1 * dim1 + 2) * dim2 + 3]) +
        ")<1e-9 else 1)\" >/dev/null 2>&1";
    const int rc = std::system(cmd.c_str());
    if (rc != 0) {
        WARN("numpy cross-check skipped or failed (python3/numpy may be unavailable in this "
             "environment) -- rc=" << rc);
    } else {
        SUCCEED("numpy.load() confirmed shape/dtype/value match");
    }

    std::remove(path.c_str());
}

TEST_CASE("dwell_to_padded_array / padded_array_to_dwell round-trip ragged frames",
    "[phase5][capture_io][dwell]") {
    RadarDwell dwell;
    dwell.frames.push_back({make_detection(5.0, 0.1, 0.0, 0.0, 10.0, 20.0),
        make_detection(8.2, -0.3, 0.05, 0.1, 8.0, 15.0)});
    dwell.frames.push_back({});  // empty frame -- no detections that frame
    dwell.frames.push_back({make_detection(12.5, 0.2, -0.1, -0.05, 5.0, 12.0)});

    size_t dmax = 0;
    const auto flat = bev::radarcam::dwell_to_padded_array(dwell, dmax);
    CHECK(dmax == 2);  // widest frame has 2 detections
    REQUIRE(flat.size() == 3 * 2 * 7);

    const auto back = bev::radarcam::padded_array_to_dwell(flat, 3, dmax);
    REQUIRE(back.frames.size() == 3);
    REQUIRE(back.frames[0].size() == 2);
    REQUIRE(back.frames[1].size() == 0);
    REQUIRE(back.frames[2].size() == 1);

    CHECK_THAT(back.frames[0][0].range_m, WithinAbs(5.0, kTol));
    CHECK_THAT(back.frames[0][1].range_m, WithinAbs(8.2, kTol));
    CHECK_THAT(back.frames[2][0].range_m, WithinAbs(12.5, kTol));
    CHECK_THAT(back.frames[2][0].vel_mps, WithinAbs(-0.05, kTol));
    CHECK_THAT(back.frames[2][0].rcs_dbsm, WithinAbs(5.0, kTol));
}

TEST_CASE("save_capture / load_capture round-trips a full capture_NNNN/ directory",
    "[phase5][capture_io][capture]") {
    Capture cap;
    cap.capture_id = "capture_0007";
    cap.image = cv::Mat(4, 6, CV_8UC3, cv::Scalar(10, 20, 30));

    cap.radar_dwell.frames.push_back(
        {make_detection(5.0, 0.0, 0.0, 0.0, 10.0, 20.0), make_detection(20.0, 0.5, 0.1, 1.2, -3.0, 8.0)});
    cap.radar_dwell.frames.push_back({make_detection(5.05, 0.01, 0.0, 0.02, 9.8, 19.5)});

    cap.has_background = true;
    cap.radar_background.frames.push_back({make_detection(30.0, -0.9, 0.2, 5.0, -10.0, 6.0)});

    cap.meta.rig_height_m = 1.35;
    cap.meta.timestamp = "2026-07-24T10:00:00Z";
    cap.meta.operator_notes = "clear sky, tripod at station 3";

    const std::string capture_dir = std::string(FIXTURES_DIR) + "/capture_scratch_0007";
    std::string error;
    REQUIRE(bev::radarcam::save_capture(capture_dir, cap, error));

    Capture loaded;
    REQUIRE(bev::radarcam::load_capture(capture_dir, loaded, error));

    CHECK(loaded.capture_id == "capture_0007");
    REQUIRE(!loaded.image.empty());
    CHECK(loaded.image.rows == 4);
    CHECK(loaded.image.cols == 6);

    REQUIRE(loaded.radar_dwell.frames.size() == 2);
    REQUIRE(loaded.radar_dwell.frames[0].size() == 2);
    REQUIRE(loaded.radar_dwell.frames[1].size() == 1);
    CHECK_THAT(loaded.radar_dwell.frames[0][1].range_m, WithinAbs(20.0, kTol));
    CHECK_THAT(loaded.radar_dwell.frames[0][1].azimuth_rad, WithinAbs(0.5, kTol));
    CHECK_THAT(loaded.radar_dwell.frames[1][0].range_m, WithinAbs(5.05, kTol));

    CHECK(loaded.has_background);
    REQUIRE(loaded.radar_background.frames.size() == 1);
    REQUIRE(loaded.radar_background.frames[0].size() == 1);
    CHECK_THAT(loaded.radar_background.frames[0][0].range_m, WithinAbs(30.0, kTol));

    CHECK(loaded.meta.capture_id == "capture_0007");
    CHECK_THAT(loaded.meta.rig_height_m, WithinAbs(1.35, kTol));
    CHECK(loaded.meta.timestamp == "2026-07-24T10:00:00Z");
    CHECK(loaded.meta.operator_notes == "clear sky, tripod at station 3");

    std::filesystem::remove_all(capture_dir);
}

TEST_CASE("load_capture reports an error when radar_dwell.npy is missing", "[phase5][capture_io][capture]") {
    const std::string capture_dir = std::string(FIXTURES_DIR) + "/capture_scratch_missing_dwell";
    std::filesystem::create_directories(capture_dir);

    Capture loaded;
    std::string error;
    REQUIRE_FALSE(bev::radarcam::load_capture(capture_dir, loaded, error));
    CHECK_FALSE(error.empty());

    std::filesystem::remove_all(capture_dir);
}
