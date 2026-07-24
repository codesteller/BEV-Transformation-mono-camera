#pragma once

// Bit-exact C++ port of radar_4d_cli.py's RadarCANFDParser + RadarState/dispatch (Columbus FC 4D
// radar CAN-FD decode: CAN IDs 0x281-0x284, raw/detected point cloud "RDI"). Every magic number and
// scale factor below is copied verbatim from the Python reference -- do not "fix" or reinterpret
// any of them, including values that look physically odd (e.g. azimuth/elevation landing outside
// +-pi after the sign-conversion + offset). Re-sync this file by hand against radar_4d_cli.py if
// that file's decode logic ever changes -- there is no shared package linking the two.
//
// Only the RDI (raw detection) path is ported. The OD/TrackedObject path (CAN IDs 0x381/0x382)
// isn't used anywhere in the radar-camera calibration pipeline (radcam_calibplan.md works from raw
// per-frame detections, never tracked objects), so it's intentionally left out.

#include <opencv2/core.hpp>

#include <cstddef>
#include <cstdint>
#include <vector>

#include "bev/radarcam/geometry.hpp"

namespace bev {
namespace radarcam {

struct RadarDetection {
    double range_m = 0.0;
    double vel_mps = 0.0;  // Doppler velocity -- radar_4d_cli.py's field name, == spec's doppler_mps
    double azimuth_rad = 0.0;
    double elevation_rad = 0.0;
    double rcs_dbsm = 0.0;
    double snr_db = 0.0;
    bool is_stationary = false;
    cv::Point3d position;  // radar-frame Cartesian, via geometry::spherical_to_cartesian
};

class RadarCanFdParser {
public:
    struct RdiHeader {
        bool valid = false;
        int radar_id = 0;
        int frame_id = 0;
        uint32_t timestamp = 0;
        int det_number = 0;
    };

    // Ports Python's `_s16`: 16-bit raw field -> signed value. Every input here is already masked
    // to <=16 bits by the caller, so plain int arithmetic (no wraparound risk) matches Python's
    // arbitrary-precision int exactly.
    static int to_signed16(int v) { return (v & 0x8000) ? (v - 0x10000) : v; }

    static RdiHeader parse_rdi_header(const std::vector<uint8_t>& data) {
        RdiHeader h;
        if (data.size() < 13) return h;
        h.valid = true;
        h.radar_id = (data[10] >> 2) & 0x07;
        h.frame_id = (data[8] << 8) | data[9];
        h.timestamp = (static_cast<uint32_t>(data[4]) << 24) | (static_cast<uint32_t>(data[5]) << 16) |
                      (static_cast<uint32_t>(data[6]) << 8) | data[7];
        h.det_number = ((data[11] & 0x7F) << 3) | ((data[12] & 0xE0) >> 5);
        return h;
    }

    static bool parse_rdi_slot(const uint8_t* p, size_t len, RadarDetection& out) {
        if (len < 19) return false;

        const int raw_r = ((p[2] & 0x3F) << 9) | (p[3] << 1) | ((p[4] >> 7) & 1);
        out.range_m = raw_r * 0.01;

        const int raw_v = ((p[5] & 0x07) << 12) | (p[6] << 4) | ((p[7] >> 4) & 0xF);
        out.vel_mps = raw_v * 0.01 - 83.3;

        const int raw_az = ((p[8] & 1) << 15) | (p[9] << 7) | ((p[10] >> 1) & 0x7F);
        out.azimuth_rad = to_signed16(raw_az) * 0.000175 - 3.15;

        const int raw_el = ((p[12] & 7) << 13) | (p[13] << 5) | ((p[14] >> 3) & 0x1F);
        out.elevation_rad = to_signed16(raw_el) * 0.000175 - 3.15;

        out.rcs_dbsm = static_cast<double>((p[16] & 0x1F) * 2 - 40);
        out.snr_db = ((p[17] & 0xFE) >> 1) * 0.5;
        out.is_stationary = ((p[17] & 0x01) ^ 0x01) != 0;

        out.position = spherical_to_cartesian(out.range_m, out.azimuth_rad, out.elevation_rad);
        return true;
    }

    static std::vector<RadarDetection> parse_rdi_body(const std::vector<uint8_t>& data) {
        constexpr size_t kSlotBytes = 19;
        std::vector<RadarDetection> dets;
        const size_t n_slots = data.size() / kSlotBytes;
        dets.reserve(n_slots);
        for (size_t i = 0; i < n_slots; ++i) {
            RadarDetection d;
            if (parse_rdi_slot(data.data() + i * kSlotBytes, kSlotBytes, d)) {
                dets.push_back(d);
            }
        }
        return dets;
    }
};

// Multi-frame dwell reassembly, ported from radar_4d_cli.py's RadarState/dispatch: a 0x281 header
// declares how many detections to expect (det_number), spanning ceil(det_number/3) body frames
// (0x282/0x283/0x284 -- up to 3 slots of 19 bytes fit in one CAN-FD frame); the accumulated
// detections publish once that many body frames have arrived.
class RadarDwellAccumulator {
public:
    void on_header(const RadarCanFdParser::RdiHeader& hdr) {
        if (!hdr.valid) return;
        frame_id_ = hdr.frame_id;
        radar_id_ = hdr.radar_id;
        pending_.clear();
        const int n = hdr.det_number;
        expected_frames_ = (n / 3) + (n % 3 ? 1 : 0);
        if (expected_frames_ == 0) expected_frames_ = 1;  // ports Python's "... or 1" fallback
        received_frames_ = 0;
    }

    void on_body(const std::vector<uint8_t>& data) {
        const auto dets = RadarCanFdParser::parse_rdi_body(data);
        pending_.insert(pending_.end(), dets.begin(), dets.end());
        ++received_frames_;
        if (received_frames_ >= expected_frames_) {
            detections_ = pending_;
        }
    }

    const std::vector<RadarDetection>& detections() const { return detections_; }
    int frame_id() const { return frame_id_; }
    int radar_id() const { return radar_id_; }
    int expected_frames() const { return expected_frames_; }
    int received_frames() const { return received_frames_; }

private:
    std::vector<RadarDetection> pending_;
    std::vector<RadarDetection> detections_;
    int frame_id_ = 0;
    int radar_id_ = 0;
    int expected_frames_ = 1;
    int received_frames_ = 0;
};

}  // namespace radarcam
}  // namespace bev
