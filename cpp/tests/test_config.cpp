#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include "bev/radarcam/config.hpp"
#include "bev/radarcam/extrinsics_io.hpp"

#include <cstdio>
#include <fstream>

using Catch::Matchers::WithinAbs;

namespace {
constexpr double kTol = 1e-9;
}

TEST_CASE("radarcam config loads every field from YAML", "[phase1][config]") {
    bev::radarcam::RadarCamConfig cfg;
    std::string error;
    const std::string path = std::string(FIXTURES_DIR) + "/sample_radarcam_config.yaml";

    REQUIRE(bev::radarcam::load_radarcam_config_yaml(path, cfg, error));
    CAPTURE(error);

    CHECK(cfg.board.inner_corners_x == 8);
    CHECK(cfg.board.inner_corners_y == 6);
    CHECK_THAT(cfg.board.square_size_m, WithinAbs(0.035, kTol));

    CHECK_THAT(cfg.x_b_measured_m[2], WithinAbs(0.12, kTol));

    CHECK_THAT(cfg.coarse_euler_deg[1], WithinAbs(-5.0, kTol));
    CHECK_THAT(cfg.coarse_euler_deg[2], WithinAbs(90.0, kTol));
    CHECK_THAT(cfg.coarse_translation_m[0], WithinAbs(0.3, kTol));

    CHECK_THAT(cfg.pnp_reproj_rms_max_px, WithinAbs(1.0, kTol));
    CHECK_THAT(cfg.board_min_tilt_deg, WithinAbs(15.0, kTol));

    CHECK_THAT(cfg.doppler_threshold_mps, WithinAbs(0.5, kTol));
    CHECK_THAT(cfg.sigma_trihedral_dbsm, WithinAbs(22.5, kTol));
    CHECK_THAT(cfg.r_ref_m, WithinAbs(1.0, kTol));
    CHECK_THAT(cfg.rcs_margin_db, WithinAbs(10.0, kTol));
    CHECK_THAT(cfg.background_match_radius_m, WithinAbs(0.3, kTol));
    CHECK_THAT(cfg.radar_height_above_ground_m, WithinAbs(0.5, kTol));
    CHECK_THAT(cfg.persistence_frac, WithinAbs(0.8, kTol));
    CHECK_THAT(cfg.spread_max_m, WithinAbs(0.10, kTol));

    REQUIRE(cfg.gate_radii_m.size() == 3);
    CHECK_THAT(cfg.gate_radii_m[0], WithinAbs(2.0, kTol));
    CHECK_THAT(cfg.gate_radii_m[1], WithinAbs(0.5, kTol));
    CHECK_THAT(cfg.gate_radii_m[2], WithinAbs(0.4, kTol));

    CHECK_THAT(cfg.ransac_inlier_threshold_m, WithinAbs(0.25, kTol));
    CHECK(cfg.max_gate_iterations == 3);
    CHECK_THAT(cfg.inlier_drop_warn_frac, WithinAbs(0.5, kTol));

    CHECK(cfg.refine_mode == "B");
    CHECK(cfg.run_mode_b_diagnostic == true);
    CHECK_THAT(cfg.huber_delta_m, WithinAbs(0.1, kTol));

    CHECK_THAT(cfg.holdout_fraction, WithinAbs(0.2, kTol));
    CHECK_THAT(cfg.svd_ratio_warn_threshold, WithinAbs(0.15, kTol));
}

TEST_CASE("radarcam config missing file reports an error, not a crash", "[phase1][config]") {
    bev::radarcam::RadarCamConfig cfg;
    std::string error;
    REQUIRE_FALSE(bev::radarcam::load_radarcam_config_yaml("/nonexistent/path.yaml", cfg, error));
    CHECK_FALSE(error.empty());
}

TEST_CASE("radarcam config partial YAML keeps defaults for missing keys", "[phase1][config]") {
    const std::string path = std::string(FIXTURES_DIR) + "/partial_config_scratch.yaml";
    {
        std::ofstream fs(path, std::ios::trunc);
        fs << "doppler_threshold_mps: 0.75\n";
    }

    bev::radarcam::RadarCamConfig cfg;  // struct defaults
    std::string error;
    REQUIRE(bev::radarcam::load_radarcam_config_yaml(path, cfg, error));

    CHECK_THAT(cfg.doppler_threshold_mps, WithinAbs(0.75, kTol));
    CHECK_THAT(cfg.rcs_margin_db, WithinAbs(10.0, kTol));  // untouched -- keeps struct default
    REQUIRE(cfg.gate_radii_m.size() == 3);
    CHECK_THAT(cfg.gate_radii_m[0], WithinAbs(2.0, kTol));

    std::remove(path.c_str());
}

TEST_CASE("ExtrinsicsData round-trips through save/load", "[phase1][extrinsics_io]") {
    bev::radarcam::ExtrinsicsData data;
    data.R = cv::Matx33d(0.0, -1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0);
    data.t = cv::Vec3d(0.12, -0.05, 1.30);
    data.euler_deg = cv::Vec3d(1.2, -3.4, 90.0);
    data.n_inliers = 27;
    data.n_total = 32;
    data.rms_residual_m = 0.021;
    data.rms_reprojection_px = 0.84;
    data.x_b_measured_m = cv::Vec3d(0.0, 0.0, 0.12);
    data.x_b_estimated_m = cv::Vec3d(0.002, -0.001, 0.115);
    data.x_b_drift_m = 0.0055;

    const char* stage_names[7] = {"background_subtraction", "spatial_gate", "doppler_gate", "rcs_gate",
        "ground_plausibility", "best_in_gate", "persistence_aggregation"};
    for (size_t i = 0; i < data.funnel.size(); ++i) {
        data.funnel[i].stage_name = stage_names[i];
        data.funnel[i].n_entering = static_cast<int>(100 - i * 10);
        data.funnel[i].n_surviving = static_cast<int>(90 - i * 10);
    }

    const std::string path = std::string(FIXTURES_DIR) + "/extrinsics_roundtrip_scratch.yaml";
    std::string error;
    REQUIRE(bev::radarcam::save_extrinsics_yaml(path, data, error));

    bev::radarcam::ExtrinsicsData loaded;
    REQUIRE(bev::radarcam::load_extrinsics_yaml(path, loaded, error));

    for (int r = 0; r < 3; ++r) {
        for (int c = 0; c < 3; ++c) {
            CHECK_THAT(loaded.R(r, c), WithinAbs(data.R(r, c), 1e-12));
        }
    }
    for (int i = 0; i < 3; ++i) {
        CHECK_THAT(loaded.t[i], WithinAbs(data.t[i], 1e-12));
        CHECK_THAT(loaded.euler_deg[i], WithinAbs(data.euler_deg[i], 1e-12));
        CHECK_THAT(loaded.x_b_measured_m[i], WithinAbs(data.x_b_measured_m[i], 1e-12));
        CHECK_THAT(loaded.x_b_estimated_m[i], WithinAbs(data.x_b_estimated_m[i], 1e-12));
    }

    CHECK(loaded.n_inliers == data.n_inliers);
    CHECK(loaded.n_total == data.n_total);
    CHECK_THAT(loaded.rms_residual_m, WithinAbs(data.rms_residual_m, 1e-12));
    CHECK_THAT(loaded.rms_reprojection_px, WithinAbs(data.rms_reprojection_px, 1e-12));
    CHECK_THAT(loaded.x_b_drift_m, WithinAbs(data.x_b_drift_m, 1e-12));

    for (size_t i = 0; i < data.funnel.size(); ++i) {
        CHECK(loaded.funnel[i].stage_name == data.funnel[i].stage_name);
        CHECK(loaded.funnel[i].n_entering == data.funnel[i].n_entering);
        CHECK(loaded.funnel[i].n_surviving == data.funnel[i].n_surviving);
    }

    std::remove(path.c_str());
}
