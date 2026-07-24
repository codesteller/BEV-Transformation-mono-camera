#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include "bev/radarcam/clutter_filter.hpp"

using Catch::Matchers::WithinAbs;
using bev::radarcam::ClutterFilterParams;
using bev::radarcam::RadarDetection;
using bev::radarcam::RadarDwell;
using bev::radarcam::run_clutter_pipeline;

namespace {
constexpr double kTol = 1e-9;

RadarDetection make_det(double range_m, double az, double el, double doppler, double rcs) {
    RadarDetection d;
    d.range_m = range_m;
    d.azimuth_rad = az;
    d.elevation_rad = el;
    d.vel_mps = doppler;
    d.rcs_dbsm = rcs;
    d.snr_db = 15.0;
    d.position = bev::radarcam::spherical_to_cartesian(range_m, az, el);
    return d;
}

// Convenience: a detection placed at an exact Cartesian point (bypassing spherical construction),
// used where the test cares about position/RCS/doppler directly rather than raw radar angles.
RadarDetection make_det_at(cv::Point3d pos, double doppler, double rcs) {
    RadarDetection d;
    d.position = pos;
    d.range_m = cv::norm(pos);
    d.vel_mps = doppler;
    d.rcs_dbsm = rcs;
    d.snr_db = 15.0;
    return d;
}

RadarDwell empty_background() { return RadarDwell{}; }
}  // namespace

TEST_CASE("stage 0: background subtraction removes matched clutter, keeps unmatched target",
    "[phase7][clutter_filter]") {
    const cv::Point3d target_pos(10.0, 0.0, 0.0);
    const cv::Point3d clutter_pos(15.0, 3.0, 0.0);  // far from target, matches a background point

    RadarDwell dwell;
    for (int i = 0; i < 2; ++i) {
        dwell.frames.push_back(
            {make_det_at(target_pos, 0.0, -20.0), make_det_at(clutter_pos, 0.0, -20.0)});
    }
    RadarDwell background;
    background.frames.push_back({make_det_at(clutter_pos + cv::Point3d(0.05, -0.02, 0.0), 0.0, -20.0)});

    ClutterFilterParams params;
    params.gate_radius_m = 2.0;
    params.background_match_radius_m = 0.3;

    const auto result = run_clutter_pipeline(dwell, background, /*has_background=*/true, target_pos, params);

    CHECK(result.funnel[0].n_entering == 4);
    CHECK(result.funnel[0].n_surviving == 2);  // clutter removed from both frames, target kept
}

TEST_CASE("stage 1: spatial gate accepts within radius, rejects just outside, tightens correctly",
    "[phase7][clutter_filter]") {
    const cv::Point3d q_hat(10.0, 0.0, 0.0);
    const cv::Point3d inside(11.5, 0.0, 0.0);   // 1.5m from q_hat
    const cv::Point3d outside(12.5, 0.0, 0.0);  // 2.5m from q_hat

    RadarDwell dwell;
    dwell.frames.push_back({make_det_at(inside, 0.0, -20.0), make_det_at(outside, 0.0, -20.0)});

    ClutterFilterParams wide;
    wide.gate_radius_m = 2.0;
    const auto wide_result = run_clutter_pipeline(dwell, empty_background(), false, q_hat, wide);
    CHECK(wide_result.funnel[1].n_entering == 2);
    CHECK(wide_result.funnel[1].n_surviving == 1);  // only `inside` (1.5m < 2.0m)

    ClutterFilterParams tight;
    tight.gate_radius_m = 0.4;
    const auto tight_result = run_clutter_pipeline(dwell, empty_background(), false, q_hat, tight);
    CHECK(tight_result.funnel[1].n_entering == 2);
    CHECK(tight_result.funnel[1].n_surviving == 0);  // both now exceed the tightened 0.4m gate
}

TEST_CASE("stage 2: Doppler gate keeps a static reflector, rejects vehicle-speed clutter",
    "[phase7][clutter_filter]") {
    const cv::Point3d q_hat(10.0, 0.0, 0.0);
    RadarDwell dwell;
    dwell.frames.push_back(
        {make_det_at(q_hat, 0.0, -20.0), make_det_at(q_hat, 8.0, -20.0)});  // 8 m/s -- a passing vehicle

    ClutterFilterParams params;
    params.gate_radius_m = 2.0;
    params.doppler_threshold_mps = 0.5;
    const auto result = run_clutter_pipeline(dwell, empty_background(), false, q_hat, params);

    CHECK(result.funnel[2].n_entering == 2);
    CHECK(result.funnel[2].n_surviving == 1);
}

TEST_CASE("stage 3: RCS gate is two-sided -- rejects both dimmer and brighter than expected",
    "[phase7][clutter_filter]") {
    const cv::Point3d q_hat(10.0, 0.0, 0.0);  // expected_dbsm = 20 - 40*log10(10/1) = -20
    RadarDwell dwell;
    dwell.frames.push_back({
        make_det_at(q_hat, 0.0, -20.0),  // matches exactly -- survives
        make_det_at(q_hat, 0.0, -5.0),   // 15 dB brighter than the +-10 dB margin allows -- rejected
        make_det_at(q_hat, 0.0, -35.0),  // 15 dB dimmer -- also rejected
    });

    ClutterFilterParams params;
    params.gate_radius_m = 2.0;
    params.sigma_trihedral_dbsm = 20.0;
    params.r_ref_m = 1.0;
    params.rcs_margin_db = 10.0;
    const auto result = run_clutter_pipeline(dwell, empty_background(), false, q_hat, params);

    CHECK(result.funnel[3].n_entering == 3);
    CHECK(result.funnel[3].n_surviving == 1);
}

TEST_CASE("stage 4: ground-plane plausibility rejects a mirrored-elevation multipath ghost",
    "[phase7][clutter_filter]") {
    const double r = 5.0;
    const auto real_target = make_det(r, 0.0, 0.1, 0.0, -7.96);       // z = 5*sin(0.1) ~ +0.5, above ground
    const auto ghost = make_det(r, 0.0, -0.3, 0.0, -7.96);            // z = 5*sin(-0.3) ~ -1.48, below ground

    RadarDwell dwell;
    dwell.frames.push_back({real_target, ghost});

    const cv::Point3d q_hat(5.0, 0.0, 0.0);
    ClutterFilterParams params;
    params.gate_radius_m = 3.0;
    params.sigma_trihedral_dbsm = 20.0;
    params.r_ref_m = 1.0;
    params.rcs_margin_db = 10.0;
    params.radar_height_above_ground_m = 0.5;
    const auto result = run_clutter_pipeline(dwell, empty_background(), false, q_hat, params);

    CHECK(result.funnel[4].n_entering == 2);
    CHECK(result.funnel[4].n_surviving == 1);
}

TEST_CASE("stage 5: best-in-gate picks the higher-RCS mainlobe, not the geometrically closer sidelobe",
    "[phase7][clutter_filter]") {
    const cv::Point3d q_hat(10.0, 0.0, 0.0);
    const cv::Point3d sidelobe_pos(10.05, 0.0, 0.0);  // only 0.05m from prediction
    const cv::Point3d mainlobe_pos(10.3, 0.0, 0.0);   // 0.3m from prediction -- farther, but the real target

    RadarDwell dwell;
    dwell.frames.push_back(
        {make_det_at(sidelobe_pos, 0.0, -25.0), make_det_at(mainlobe_pos, 0.0, -20.0)});

    ClutterFilterParams params;
    params.gate_radius_m = 2.0;
    params.sigma_trihedral_dbsm = 20.0;
    params.r_ref_m = 1.0;
    params.rcs_margin_db = 10.0;
    // Single frame, so persistence (1/1) must exceed persistence_frac -- lower it for this test since
    // it's specifically about stage 5's selection, not stage 6's persistence math.
    params.persistence_frac = 0.0;
    params.spread_max_m = 1.0;

    const auto result = run_clutter_pipeline(dwell, empty_background(), false, q_hat, params);

    REQUIRE(result.accepted);
    CHECK_THAT(result.q_radar.x, WithinAbs(mainlobe_pos.x, kTol));  // picked the mainlobe, not the sidelobe
    CHECK_THAT(result.detection_rcs_dbsm, WithinAbs(-20.0, kTol));
}

TEST_CASE("stage 6: high persistence + tight spread accepts with the correct median",
    "[phase7][clutter_filter]") {
    const cv::Point3d q_hat(10.0, 0.0, 0.0);
    RadarDwell dwell;
    for (int i = 0; i < 25; ++i) {
        dwell.frames.push_back({make_det_at(q_hat, 0.0, -20.0)});
    }
    for (int i = 0; i < 5; ++i) {
        dwell.frames.push_back({});  // missed detection this frame
    }

    ClutterFilterParams params;
    params.gate_radius_m = 2.0;
    params.sigma_trihedral_dbsm = 20.0;
    params.r_ref_m = 1.0;
    params.rcs_margin_db = 10.0;
    params.persistence_frac = 0.8;
    params.spread_max_m = 0.10;

    const auto result = run_clutter_pipeline(dwell, empty_background(), false, q_hat, params);

    REQUIRE(result.accepted);
    CHECK_THAT(result.q_radar.x, WithinAbs(10.0, kTol));
    CHECK_THAT(result.dwell_spread_m, WithinAbs(0.0, kTol));
    CHECK_THAT(result.detection_rcs_dbsm, WithinAbs(-20.0, kTol));

    // Exact funnel counts, hand-computed: 25 raw detections, all pass stages 0-4 trivially; stage 5
    // yields one per-frame-best per non-empty frame (25); stage 6 sees 25/30 = 0.833 > 0.8, accepts.
    CHECK(result.funnel[0].n_entering == 25);
    CHECK(result.funnel[0].n_surviving == 25);
    CHECK(result.funnel[1].n_entering == 25);
    CHECK(result.funnel[1].n_surviving == 25);
    CHECK(result.funnel[2].n_entering == 25);
    CHECK(result.funnel[2].n_surviving == 25);
    CHECK(result.funnel[3].n_entering == 25);
    CHECK(result.funnel[3].n_surviving == 25);
    CHECK(result.funnel[4].n_entering == 25);
    CHECK(result.funnel[4].n_surviving == 25);
    CHECK(result.funnel[5].n_entering == 25);
    CHECK(result.funnel[5].n_surviving == 25);
    CHECK(result.funnel[6].n_entering == 25);
    CHECK(result.funnel[6].n_surviving == 25);
}

TEST_CASE("stage 6: low persistence rejects the capture", "[phase7][clutter_filter]") {
    const cv::Point3d q_hat(10.0, 0.0, 0.0);
    RadarDwell dwell;
    for (int i = 0; i < 10; ++i) dwell.frames.push_back({make_det_at(q_hat, 0.0, -20.0)});
    for (int i = 0; i < 20; ++i) dwell.frames.push_back({});

    ClutterFilterParams params;
    params.gate_radius_m = 2.0;
    params.sigma_trihedral_dbsm = 20.0;
    params.r_ref_m = 1.0;
    params.rcs_margin_db = 10.0;
    params.persistence_frac = 0.8;
    params.spread_max_m = 0.10;

    const auto result = run_clutter_pipeline(dwell, empty_background(), false, q_hat, params);

    CHECK_FALSE(result.accepted);
    CHECK(result.reject_reason.find("persistence") != std::string::npos);
}

TEST_CASE("stage 6: high persistence but excessive spread rejects the capture",
    "[phase7][clutter_filter]") {
    const cv::Point3d q_hat(10.0, 0.0, 0.0);
    RadarDwell dwell;
    for (int i = 0; i < 29; ++i) dwell.frames.push_back({make_det_at(q_hat, 0.0, -20.0)});
    dwell.frames.push_back({make_det_at(cv::Point3d(10.5, 0.0, 0.0), 0.0, -20.0)});  // one outlier frame

    ClutterFilterParams params;
    params.gate_radius_m = 2.0;
    params.sigma_trihedral_dbsm = 20.0;
    params.r_ref_m = 1.0;
    params.rcs_margin_db = 10.0;
    params.persistence_frac = 0.8;
    params.spread_max_m = 0.10;

    const auto result = run_clutter_pipeline(dwell, empty_background(), false, q_hat, params);

    CHECK_FALSE(result.accepted);
    CHECK(result.reject_reason.find("spread") != std::string::npos);
}
