#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include "bev/radarcam/can_decoder.hpp"
#include "fixtures/can_decoder_golden.hpp"

using Catch::Matchers::WithinAbs;
using bev::radarcam::RadarCanFdParser;
using bev::radarcam::RadarDetection;
using bev::radarcam::RadarDwellAccumulator;
namespace golden = bev::radarcam::test_fixtures;

namespace {
constexpr double kTol = 1e-9;
}

TEST_CASE("parse_rdi_slot matches the Python reference decoder, typical stationary case",
    "[phase3][can_decoder]") {
    RadarDetection d;
    REQUIRE(RadarCanFdParser::parse_rdi_slot(
        golden::kSlotV1TypicalStationary.data(), golden::kSlotV1TypicalStationary.size(), d));

    CHECK_THAT(d.range_m, WithinAbs(5.0, kTol));
    CHECK_THAT(d.vel_mps, WithinAbs(0.0, kTol));
    CHECK_THAT(d.azimuth_rad, WithinAbs(0.0, kTol));
    CHECK_THAT(d.elevation_rad, WithinAbs(0.0, kTol));
    CHECK_THAT(d.rcs_dbsm, WithinAbs(0.0, kTol));
    CHECK_THAT(d.snr_db, WithinAbs(5.0, kTol));
    CHECK(d.is_stationary == true);

    CHECK_THAT(d.position.x, WithinAbs(5.0, kTol));
    CHECK_THAT(d.position.y, WithinAbs(0.0, kTol));
    CHECK_THAT(d.position.z, WithinAbs(0.0, kTol));
}

TEST_CASE("parse_rdi_slot matches the Python reference decoder, sign-bit-set case",
    "[phase3][can_decoder]") {
    RadarDetection d;
    REQUIRE(RadarCanFdParser::parse_rdi_slot(
        golden::kSlotV2SignBitSet.data(), golden::kSlotV2SignBitSet.size(), d));

    CHECK_THAT(d.range_m, WithinAbs(12.34, kTol));
    CHECK_THAT(d.vel_mps, WithinAbs(-33.3, 1e-8));
    CHECK_THAT(d.azimuth_rad, WithinAbs(-7.6188, 1e-8));
    CHECK_THAT(d.elevation_rad, WithinAbs(-5.8688, 1e-8));
    CHECK_THAT(d.rcs_dbsm, WithinAbs(22.0, kTol));
    CHECK_THAT(d.snr_db, WithinAbs(50.0, kTol));
    CHECK(d.is_stationary == false);

    CHECK_THAT(d.position.x, WithinAbs(2.632094495484701, 1e-9));
    CHECK_THAT(d.position.y, WithinAbs(-10.9846469104106, 1e-9));
    CHECK_THAT(d.position.z, WithinAbs(4.968421361000493, 1e-9));
}

TEST_CASE("parse_rdi_slot matches the Python reference decoder, boundary/extreme case",
    "[phase3][can_decoder]") {
    RadarDetection d;
    REQUIRE(RadarCanFdParser::parse_rdi_slot(
        golden::kSlotV3BoundaryExtremes.data(), golden::kSlotV3BoundaryExtremes.size(), d));

    CHECK_THAT(d.range_m, WithinAbs(327.67, kTol));
    CHECK_THAT(d.vel_mps, WithinAbs(-83.3, 1e-8));
    CHECK_THAT(d.azimuth_rad, WithinAbs(-3.15, kTol));
    CHECK_THAT(d.elevation_rad, WithinAbs(-3.15, kTol));
    CHECK_THAT(d.rcs_dbsm, WithinAbs(-40.0, kTol));
    CHECK_THAT(d.snr_db, WithinAbs(0.0, kTol));
    CHECK(d.is_stationary == true);

    CHECK_THAT(d.position.x, WithinAbs(327.64683969187683, 1e-9));
    CHECK_THAT(d.position.y, WithinAbs(-2.754705385853436, 1e-9));
    CHECK_THAT(d.position.z, WithinAbs(2.754802744793588, 1e-9));
}

TEST_CASE("parse_rdi_slot rejects payloads shorter than one slot", "[phase3][can_decoder]") {
    std::vector<uint8_t> short_payload(18, 0);
    RadarDetection d;
    CHECK_FALSE(RadarCanFdParser::parse_rdi_slot(short_payload.data(), short_payload.size(), d));
}

TEST_CASE("parse_rdi_header matches the Python reference decoder", "[phase3][can_decoder]") {
    const auto hdr = RadarCanFdParser::parse_rdi_header(golden::kHeaderDetNumber5);
    REQUIRE(hdr.valid);
    CHECK(hdr.radar_id == 2);
    CHECK(hdr.frame_id == 7);
    CHECK(hdr.timestamp == 1u);
    CHECK(hdr.det_number == 5);
}

TEST_CASE("parse_rdi_header rejects payloads shorter than the header", "[phase3][can_decoder]") {
    const std::vector<uint8_t> short_payload(12, 0);
    const auto hdr = RadarCanFdParser::parse_rdi_header(short_payload);
    CHECK_FALSE(hdr.valid);
}

TEST_CASE("dwell reassembly: det_number=5 spans two body frames (3+2), matching Python",
    "[phase3][can_decoder][reassembly]") {
    RadarDwellAccumulator acc;

    acc.on_header(RadarCanFdParser::parse_rdi_header(golden::kHeaderDetNumber5));
    CHECK(acc.expected_frames() == 2);
    CHECK(acc.received_frames() == 0);
    CHECK(acc.detections().empty());

    std::vector<uint8_t> body1;
    body1.insert(body1.end(), golden::kSlotV1TypicalStationary.begin(), golden::kSlotV1TypicalStationary.end());
    body1.insert(body1.end(), golden::kSlotV2SignBitSet.begin(), golden::kSlotV2SignBitSet.end());
    body1.insert(body1.end(), golden::kSlotV3BoundaryExtremes.begin(), golden::kSlotV3BoundaryExtremes.end());
    acc.on_body(body1);
    CHECK(acc.received_frames() == 1);
    CHECK(acc.detections().empty());  // not yet published -- only 1/2 expected frames arrived

    std::vector<uint8_t> body2;
    body2.insert(body2.end(), golden::kSlotV1TypicalStationary.begin(), golden::kSlotV1TypicalStationary.end());
    body2.insert(body2.end(), golden::kSlotV2SignBitSet.begin(), golden::kSlotV2SignBitSet.end());
    acc.on_body(body2);
    CHECK(acc.received_frames() == 2);
    REQUIRE(acc.detections().size() == 5);  // 3 + 2, published now that 2/2 frames arrived
}

TEST_CASE("dwell reassembly: det_number=0 falls back to expecting 1 frame ('... or 1' in Python)",
    "[phase3][can_decoder][reassembly]") {
    RadarDwellAccumulator acc;
    acc.on_header(RadarCanFdParser::parse_rdi_header(golden::kHeaderDetNumber0));
    CHECK(acc.expected_frames() == 1);

    acc.on_body({});  // empty body, 0 slots
    CHECK(acc.received_frames() == 1);
    CHECK(acc.detections().empty());  // published (1/1 frames arrived), just an empty list
}

TEST_CASE("dwell reassembly: det_number=6 spans exactly two full 3-slot frames, no remainder",
    "[phase3][can_decoder][reassembly]") {
    RadarDwellAccumulator acc;
    acc.on_header(RadarCanFdParser::parse_rdi_header(golden::kHeaderDetNumber6));
    CHECK(acc.expected_frames() == 2);

    std::vector<uint8_t> body_a;
    for (int i = 0; i < 3; ++i) {
        body_a.insert(body_a.end(), golden::kSlotV1TypicalStationary.begin(), golden::kSlotV1TypicalStationary.end());
    }
    acc.on_body(body_a);
    CHECK(acc.received_frames() == 1);
    CHECK(acc.detections().empty());

    std::vector<uint8_t> body_b;
    for (int i = 0; i < 3; ++i) {
        body_b.insert(body_b.end(), golden::kSlotV2SignBitSet.begin(), golden::kSlotV2SignBitSet.end());
    }
    acc.on_body(body_b);
    CHECK(acc.received_frames() == 2);
    REQUIRE(acc.detections().size() == 6);
}
