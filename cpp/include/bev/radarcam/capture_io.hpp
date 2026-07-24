#pragma once

// Reads/writes the capture_NNNN/ directory format from radcam_calibplan.md §3.1:
//   capture_0001/
//     image.png          -- camera frame, target present
//     radar_dwell.npy     -- (F, Dmax, 7) float64, F frames x zero-padded max detections x columns
//     radar_bg.npy         -- optional, same shape, background/no-target dwell
//     meta.yaml             -- rig height, timestamp, operator notes (yaml-cpp; spec's literal
//                              "meta.json" was replaced with yaml since yaml-cpp is already a
//                              dependency of this tool -- see Phase 1's config.hpp)
//
// Column order per §3.1: [range_m, azimuth_rad, elevation_rad, doppler_mps, rcs_dbsm, snr_db, valid].
// Frames are ragged (different detection counts); padding rows get valid=0 so the array stays
// rectangular. `is_stationary` isn't part of this on-disk schema -- it's not one of the spec's 7
// columns, and the clutter filter's Doppler gate (§6) derives staticness straight from doppler_mps,
// so nothing downstream needs it round-tripped through disk.
//
// §10 requires logging every raw frame (not just the post-aggregation median) so analysis can be
// re-run without re-capturing -- hence writing the full (F, Dmax, 7) dwell here, never a collapsed
// single detection per capture.

#include <yaml-cpp/yaml.h>

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include "bev/radarcam/can_decoder.hpp"

namespace bev {
namespace radarcam {

// One dwell: a sequence of frames, each holding zero or more radar detections. Ragged by
// construction -- different frames can carry different detection counts.
struct RadarDwell {
    std::vector<std::vector<RadarDetection>> frames;
};

struct CaptureMeta {
    std::string capture_id;
    double rig_height_m = 0.0;
    std::string timestamp;
    std::string operator_notes;
};

struct Capture {
    std::string capture_id;
    cv::Mat image;
    RadarDwell radar_dwell;
    RadarDwell radar_background;
    bool has_background = false;
    CaptureMeta meta;
};

// ---------------------------------------------------------------------------------------------
// Hand-rolled real .npy (NumPy format v1.0) writer/reader for a rectangular (dim0, dim1, dim2)
// float64 array, so any capture can be re-analyzed from Python with a plain `numpy.load()` -- no
// numpy/Python dependency to write or read it from C++. Byte layout verified against genuine
// numpy.save() output (magic + version(1,0) + 2-byte little-endian header length + a
// space-padded, newline-terminated ASCII header dict, aligned so the whole prefix is a multiple
// of 64 bytes -- numpy's own ARRAY_ALIGN), then raw little-endian float64 data in C (row-major)
// order. Assumes a little-endian host, matching this repo's target platforms.
// ---------------------------------------------------------------------------------------------

inline bool write_npy_f64_3d(const std::string& path, const std::vector<double>& flat_row_major,
    size_t dim0, size_t dim1, size_t dim2, std::string& error) {
    if (flat_row_major.size() != dim0 * dim1 * dim2) {
        error = "flat array size does not match dim0*dim1*dim2.";
        return false;
    }

    std::ostringstream header_stream;
    header_stream << "{'descr': '<f8', 'fortran_order': False, 'shape': (" << dim0 << ", " << dim1
                  << ", " << dim2 << "), }";
    const std::string core_header = header_stream.str();

    const size_t hlen = core_header.size() + 1;  // +1 accounts for the trailing '\n'
    constexpr size_t kMagicPlusVersionLen = 8;    // 6-byte magic + 2-byte version
    constexpr size_t kLenFieldSize = 2;
    constexpr size_t kAlign = 64;
    const size_t padlen = kAlign - ((kMagicPlusVersionLen + kLenFieldSize + hlen) % kAlign);
    const uint16_t header_len_field = static_cast<uint16_t>(hlen + padlen);

    std::ofstream fs(path, std::ios::out | std::ios::binary | std::ios::trunc);
    if (!fs.is_open()) {
        error = "Could not open file for writing: " + path;
        return false;
    }

    fs.write("\x93NUMPY", 6);
    const char version[2] = {1, 0};
    fs.write(version, 2);
    fs.write(reinterpret_cast<const char*>(&header_len_field), sizeof(header_len_field));
    fs.write(core_header.data(), static_cast<std::streamsize>(core_header.size()));
    const std::string padding(padlen, ' ');
    fs.write(padding.data(), static_cast<std::streamsize>(padding.size()));
    fs.write("\n", 1);
    fs.write(reinterpret_cast<const char*>(flat_row_major.data()),
        static_cast<std::streamsize>(flat_row_major.size() * sizeof(double)));

    return fs.good();
}

inline bool read_npy_f64_3d(const std::string& path, std::vector<double>& flat_row_major, size_t& dim0,
    size_t& dim1, size_t& dim2, std::string& error) {
    std::ifstream fs(path, std::ios::in | std::ios::binary);
    if (!fs.is_open()) {
        error = "Could not open file for reading: " + path;
        return false;
    }

    char magic[6];
    fs.read(magic, 6);
    if (!fs || std::memcmp(magic, "\x93NUMPY", 6) != 0) {
        error = "Not a valid .npy file (bad magic).";
        return false;
    }
    unsigned char version[2];
    fs.read(reinterpret_cast<char*>(version), 2);

    uint16_t header_len = 0;
    fs.read(reinterpret_cast<char*>(&header_len), sizeof(header_len));
    if (!fs) {
        error = "Truncated .npy header.";
        return false;
    }

    std::string header(header_len, '\0');
    fs.read(&header[0], static_cast<std::streamsize>(header_len));
    if (!fs) {
        error = "Truncated .npy header content.";
        return false;
    }

    const size_t shape_pos = header.find("'shape':");
    if (shape_pos == std::string::npos) {
        error = "Could not find 'shape' in .npy header.";
        return false;
    }
    const size_t open_paren = header.find('(', shape_pos);
    const size_t close_paren = (open_paren == std::string::npos) ? std::string::npos : header.find(')', open_paren);
    if (open_paren == std::string::npos || close_paren == std::string::npos) {
        error = "Malformed shape tuple in .npy header.";
        return false;
    }

    std::vector<size_t> dims;
    std::stringstream ss(header.substr(open_paren + 1, close_paren - open_paren - 1));
    std::string token;
    while (std::getline(ss, token, ',')) {
        const size_t first = token.find_first_not_of(" \t");
        if (first == std::string::npos) continue;
        const size_t last = token.find_last_not_of(" \t");
        dims.push_back(static_cast<size_t>(std::stoul(token.substr(first, last - first + 1))));
    }
    if (dims.size() != 3) {
        error = "Expected a 3-D shape in .npy header, got " + std::to_string(dims.size()) + " dims.";
        return false;
    }
    dim0 = dims[0];
    dim1 = dims[1];
    dim2 = dims[2];

    const size_t n_values = dim0 * dim1 * dim2;
    flat_row_major.assign(n_values, 0.0);
    fs.read(reinterpret_cast<char*>(flat_row_major.data()), static_cast<std::streamsize>(n_values * sizeof(double)));
    if (!fs) {
        error = "Truncated .npy data section.";
        return false;
    }
    return true;
}

// ---------------------------------------------------------------------------------------------
// RadarDwell <-> padded (F, Dmax, 7) flat array conversion.
// ---------------------------------------------------------------------------------------------

inline std::vector<double> dwell_to_padded_array(const RadarDwell& dwell, size_t& dmax) {
    dmax = 0;
    for (const auto& frame : dwell.frames) dmax = std::max(dmax, frame.size());

    const size_t f = dwell.frames.size();
    std::vector<double> flat(f * dmax * 7, 0.0);  // zero-init -- padding rows keep valid=0
    for (size_t fi = 0; fi < f; ++fi) {
        const auto& frame = dwell.frames[fi];
        for (size_t di = 0; di < frame.size(); ++di) {
            const auto& d = frame[di];
            const size_t base = (fi * dmax + di) * 7;
            flat[base + 0] = d.range_m;
            flat[base + 1] = d.azimuth_rad;
            flat[base + 2] = d.elevation_rad;
            flat[base + 3] = d.vel_mps;  // spec's doppler_mps
            flat[base + 4] = d.rcs_dbsm;
            flat[base + 5] = d.snr_db;
            flat[base + 6] = 1.0;  // valid
        }
    }
    return flat;
}

inline RadarDwell padded_array_to_dwell(const std::vector<double>& flat, size_t f, size_t dmax) {
    RadarDwell dwell;
    dwell.frames.resize(f);
    for (size_t fi = 0; fi < f; ++fi) {
        for (size_t di = 0; di < dmax; ++di) {
            const size_t base = (fi * dmax + di) * 7;
            if (flat[base + 6] == 0.0) continue;  // padding slot, not a real detection
            RadarDetection d;
            d.range_m = flat[base + 0];
            d.azimuth_rad = flat[base + 1];
            d.elevation_rad = flat[base + 2];
            d.vel_mps = flat[base + 3];
            d.rcs_dbsm = flat[base + 4];
            d.snr_db = flat[base + 5];
            d.position = spherical_to_cartesian(d.range_m, d.azimuth_rad, d.elevation_rad);
            dwell.frames[fi].push_back(d);
        }
    }
    return dwell;
}

// ---------------------------------------------------------------------------------------------
// meta.yaml
// ---------------------------------------------------------------------------------------------

inline bool save_capture_meta_yaml(const std::string& path, const CaptureMeta& meta, std::string& error) {
    YAML::Emitter out;
    out << YAML::BeginMap;
    out << YAML::Key << "capture_id" << YAML::Value << meta.capture_id;
    out << YAML::Key << "rig_height_m" << YAML::Value << meta.rig_height_m;
    out << YAML::Key << "timestamp" << YAML::Value << meta.timestamp;
    out << YAML::Key << "operator_notes" << YAML::Value << meta.operator_notes;
    out << YAML::EndMap;

    std::ofstream fs(path, std::ios::out | std::ios::trunc);
    if (!fs.is_open()) {
        error = "Could not open file for writing: " + path;
        return false;
    }
    fs << out.c_str();
    return true;
}

inline bool load_capture_meta_yaml(const std::string& path, CaptureMeta& out, std::string& error) {
    YAML::Node root;
    try {
        root = YAML::LoadFile(path);
    } catch (const std::exception& e) {
        error = std::string("Failed to load/parse YAML: ") + e.what();
        return false;
    }
    if (root["capture_id"]) out.capture_id = root["capture_id"].as<std::string>();
    if (root["rig_height_m"]) out.rig_height_m = root["rig_height_m"].as<double>();
    if (root["timestamp"]) out.timestamp = root["timestamp"].as<std::string>();
    if (root["operator_notes"]) out.operator_notes = root["operator_notes"].as<std::string>();
    return true;
}

// ---------------------------------------------------------------------------------------------
// Whole capture_NNNN/ directory.
// ---------------------------------------------------------------------------------------------

inline bool save_capture(const std::string& capture_dir, const Capture& capture, std::string& error) {
    std::error_code ec;
    std::filesystem::create_directories(capture_dir, ec);
    if (ec) {
        error = "Could not create capture directory: " + capture_dir + " (" + ec.message() + ")";
        return false;
    }

    if (!capture.image.empty()) {
        if (!cv::imwrite(capture_dir + "/image.png", capture.image)) {
            error = "Failed to write image.png";
            return false;
        }
    }

    size_t dmax = 0;
    const auto flat = dwell_to_padded_array(capture.radar_dwell, dmax);
    if (!write_npy_f64_3d(
            capture_dir + "/radar_dwell.npy", flat, capture.radar_dwell.frames.size(), dmax, 7, error)) {
        return false;
    }

    if (capture.has_background) {
        size_t bg_dmax = 0;
        const auto bg_flat = dwell_to_padded_array(capture.radar_background, bg_dmax);
        if (!write_npy_f64_3d(capture_dir + "/radar_bg.npy", bg_flat, capture.radar_background.frames.size(),
                bg_dmax, 7, error)) {
            return false;
        }
    }

    CaptureMeta meta = capture.meta;
    meta.capture_id = capture.capture_id;
    return save_capture_meta_yaml(capture_dir + "/meta.yaml", meta, error);
}

inline bool load_capture(const std::string& capture_dir, Capture& out, std::string& error) {
    const std::string image_path = capture_dir + "/image.png";
    if (std::filesystem::exists(image_path)) {
        out.image = cv::imread(image_path);
    }

    const std::string dwell_path = capture_dir + "/radar_dwell.npy";
    if (!std::filesystem::exists(dwell_path)) {
        error = "Missing radar_dwell.npy in " + capture_dir;
        return false;
    }
    std::vector<double> flat;
    size_t f = 0, dmax = 0, cols = 0;
    if (!read_npy_f64_3d(dwell_path, flat, f, dmax, cols, error)) return false;
    if (cols != 7) {
        error = "radar_dwell.npy has unexpected column count (expected 7, got " + std::to_string(cols) + ")";
        return false;
    }
    out.radar_dwell = padded_array_to_dwell(flat, f, dmax);

    const std::string bg_path = capture_dir + "/radar_bg.npy";
    out.has_background = std::filesystem::exists(bg_path);
    if (out.has_background) {
        std::vector<double> bg_flat;
        size_t bf = 0, bdmax = 0, bcols = 0;
        if (!read_npy_f64_3d(bg_path, bg_flat, bf, bdmax, bcols, error)) return false;
        out.radar_background = padded_array_to_dwell(bg_flat, bf, bdmax);
    }

    const std::string meta_path = capture_dir + "/meta.yaml";
    if (std::filesystem::exists(meta_path)) {
        if (!load_capture_meta_yaml(meta_path, out.meta, error)) return false;
        out.capture_id = out.meta.capture_id;
    }

    return true;
}

}  // namespace radarcam
}  // namespace bev
