#pragma once

// Diagnostic plot rendering (§12 item 6), drawn directly onto cv::Mat canvases with OpenCV
// primitives and shown through the app's existing mat_to_qimage()+QLabel pattern -- explicitly not
// QCustomPlot (GPLv3, conflicts with this repo's Apache-2.0 license) or QChart. Not numerically
// tested (it's rendering, not computation); "done" for this phase is a human visual check of
// generated PNGs.

#include "bev/radarcam/clutter_filter.hpp"
#include "bev/radarcam/gate_iterate.hpp"
#include "bev/radarcam/kabsch.hpp"

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <string>
#include <vector>

namespace bev {
namespace radarcam {

// Palette matching the app's existing dark theme (see calibration_tool_qt.cpp's stylesheet) plus a
// validated status/categorical color set: green/red for a genuine inlier/outlier status pairing,
// two categorical hues for the two-series plots, muted gray for chrome (axes/labels), never full
// brightness except for actual data.
namespace palette {
inline const cv::Scalar kBackground(30, 30, 30);  // #1e1e1e
inline const cv::Scalar kAxis(120, 120, 120);
inline const cv::Scalar kText(210, 210, 210);
inline const cv::Scalar kGood(12, 163, 12);      // #0ca30c -- inliers
inline const cv::Scalar kCritical(59, 59, 208);  // #d03b3b -- outliers
inline const cv::Scalar kSeries1(229, 135, 57);  // #3987e5 (BGR) -- measured / PnP
inline const cv::Scalar kSeries2(38, 89, 217);   // #d95926 (BGR) -- predicted / radar
}  // namespace palette

namespace detail {

struct AxisMap {
    double min_x = 0.0, max_x = 1.0, min_y = 0.0, max_y = 1.0;
    cv::Rect plot_rect;

    cv::Point to_px(double x, double y) const {
        const double tx = (max_x > min_x) ? (x - min_x) / (max_x - min_x) : 0.5;
        const double ty = (max_y > min_y) ? (y - min_y) / (max_y - min_y) : 0.5;
        const int px = plot_rect.x + static_cast<int>(tx * plot_rect.width);
        const int py = plot_rect.y + plot_rect.height - static_cast<int>(ty * plot_rect.height);
        return cv::Point(px, py);
    }
};

inline AxisMap make_axis_map(
    const std::vector<double>& xs, const std::vector<double>& ys, const cv::Rect& plot_rect, double pad_frac = 0.1) {
    AxisMap m;
    m.plot_rect = plot_rect;
    if (xs.empty()) return m;
    const double min_x = *std::min_element(xs.begin(), xs.end());
    const double max_x = *std::max_element(xs.begin(), xs.end());
    const double min_y = *std::min_element(ys.begin(), ys.end());
    const double max_y = *std::max_element(ys.begin(), ys.end());
    const double range_x = std::max(max_x - min_x, 1e-6);
    const double range_y = std::max(max_y - min_y, 1e-6);
    m.min_x = min_x - pad_frac * range_x;
    m.max_x = max_x + pad_frac * range_x;
    m.min_y = min_y - pad_frac * range_y;
    m.max_y = max_y + pad_frac * range_y;
    return m;
}

inline void draw_axes_box(cv::Mat& canvas, const cv::Rect& plot_rect) { cv::rectangle(canvas, plot_rect, palette::kAxis, 1); }

inline void draw_legend_swatch(cv::Mat& canvas, cv::Point origin, const cv::Scalar& color, const std::string& label) {
    cv::rectangle(canvas, cv::Rect(origin.x, origin.y, 14, 14), color, cv::FILLED);
    cv::putText(
        canvas, label, cv::Point(origin.x + 20, origin.y + 12), cv::FONT_HERSHEY_SIMPLEX, 0.4, palette::kText, 1, cv::LINE_AA);
}

}  // namespace detail

inline cv::Mat make_blank_canvas(int width = 640, int height = 480) {
    return cv::Mat(height, width, CV_8UC3, palette::kBackground);
}

// Reflector position cloud, colored by inlier/outlier, as two 2D projections (top-down x/y, and
// range vs height) side by side -- avoids a fake/ambiguous 3D perspective on a static canvas.
inline cv::Mat render_reflector_cloud(
    const std::vector<cv::Point3d>& positions, const std::vector<bool>& is_inlier, int width = 800, int height = 400) {
    cv::Mat canvas = make_blank_canvas(width, height);
    if (positions.empty()) return canvas;

    const cv::Rect left_rect(40, 20, width / 2 - 60, height - 60);
    const cv::Rect right_rect(width / 2 + 20, 20, width / 2 - 60, height - 60);

    std::vector<double> xs, ys, ranges, zs;
    for (const auto& p : positions) {
        xs.push_back(p.x);
        ys.push_back(p.y);
        ranges.push_back(cv::norm(p));
        zs.push_back(p.z);
    }
    const auto map_xy = detail::make_axis_map(xs, ys, left_rect);
    const auto map_rz = detail::make_axis_map(ranges, zs, right_rect);

    detail::draw_axes_box(canvas, left_rect);
    detail::draw_axes_box(canvas, right_rect);
    cv::putText(canvas, "Top-down (x/y)", cv::Point(left_rect.x, left_rect.y - 6), cv::FONT_HERSHEY_SIMPLEX, 0.45,
        palette::kText, 1, cv::LINE_AA);
    cv::putText(canvas, "Range vs height", cv::Point(right_rect.x, right_rect.y - 6), cv::FONT_HERSHEY_SIMPLEX, 0.45,
        palette::kText, 1, cv::LINE_AA);

    for (size_t i = 0; i < positions.size(); ++i) {
        const bool inlier = (i < is_inlier.size()) ? is_inlier[i] : true;
        const cv::Scalar color = inlier ? palette::kGood : palette::kCritical;
        cv::circle(canvas, map_xy.to_px(xs[i], ys[i]), 4, color, cv::FILLED, cv::LINE_AA);
        cv::circle(canvas, map_rz.to_px(ranges[i], zs[i]), 4, color, cv::FILLED, cv::LINE_AA);
    }

    detail::draw_legend_swatch(canvas, cv::Point(width - 110, height - 30), palette::kGood, "inlier");
    detail::draw_legend_swatch(canvas, cv::Point(width - 110, height - 12), palette::kCritical, "outlier");
    return canvas;
}

// Generic single-series scatter, used for both "residual vs range" and "residual vs azimuth".
inline cv::Mat render_scatter(const std::vector<double>& x, const std::vector<double>& y, const std::string& x_label,
    const std::string& y_label, int width = 640, int height = 400) {
    cv::Mat canvas = make_blank_canvas(width, height);
    if (x.empty() || x.size() != y.size()) return canvas;

    const cv::Rect plot_rect(50, 20, width - 80, height - 60);
    const auto map = detail::make_axis_map(x, y, plot_rect);
    detail::draw_axes_box(canvas, plot_rect);

    for (size_t i = 0; i < x.size(); ++i) {
        cv::circle(canvas, map.to_px(x[i], y[i]), 4, palette::kSeries1, cv::FILLED, cv::LINE_AA);
    }

    cv::putText(canvas, x_label, cv::Point(plot_rect.x, plot_rect.y + plot_rect.height + 24), cv::FONT_HERSHEY_SIMPLEX,
        0.45, palette::kText, 1, cv::LINE_AA);
    cv::putText(
        canvas, y_label, cv::Point(6, plot_rect.y + 10), cv::FONT_HERSHEY_SIMPLEX, 0.45, palette::kText, 1, cv::LINE_AA);
    return canvas;
}

// Reprojection overlay: draws the PnP-measured reflector pixel and the radar-predicted reflector
// pixel on a copy of `base_image`, connected by a thin residual-vector line.
inline cv::Mat render_reprojection_overlay(
    const cv::Mat& base_image, const cv::Point2d& measured_px, const cv::Point2d& predicted_px) {
    cv::Mat canvas = base_image.empty() ? make_blank_canvas() : base_image.clone();
    if (canvas.channels() == 1) cv::cvtColor(canvas, canvas, cv::COLOR_GRAY2BGR);

    const cv::Point measured(static_cast<int>(measured_px.x), static_cast<int>(measured_px.y));
    const cv::Point predicted(static_cast<int>(predicted_px.x), static_cast<int>(predicted_px.y));

    cv::line(canvas, measured, predicted, palette::kText, 1, cv::LINE_AA);
    cv::circle(canvas, measured, 6, palette::kSeries1, 2, cv::LINE_AA);
    cv::circle(canvas, predicted, 6, palette::kSeries2, 2, cv::LINE_AA);

    detail::draw_legend_swatch(canvas, cv::Point(10, 10), palette::kSeries1, "measured (PnP)");
    detail::draw_legend_swatch(canvas, cv::Point(10, 28), palette::kSeries2, "predicted (radar)");
    return canvas;
}

// Inlier count vs. gate iteration -- a handful of discrete points, direct-labeled (no gridlines
// needed at this cardinality). Guardrail-triggered iterations are marked in the "critical" color.
inline cv::Mat render_inlier_vs_iteration(const std::vector<IterationRecord>& iterations, int width = 640, int height = 300) {
    cv::Mat canvas = make_blank_canvas(width, height);
    if (iterations.empty()) return canvas;

    const cv::Rect plot_rect(50, 20, width - 80, height - 60);
    std::vector<double> xs, ys;
    for (const auto& rec : iterations) {
        xs.push_back(static_cast<double>(rec.iteration));
        ys.push_back(static_cast<double>(rec.n_inliers));
    }
    const auto map = detail::make_axis_map(xs, ys, plot_rect, 0.2);
    detail::draw_axes_box(canvas, plot_rect);

    for (size_t i = 0; i + 1 < iterations.size(); ++i) {
        cv::line(canvas, map.to_px(xs[i], ys[i]), map.to_px(xs[i + 1], ys[i + 1]), palette::kSeries1, 2, cv::LINE_AA);
    }
    for (size_t i = 0; i < iterations.size(); ++i) {
        const cv::Point p = map.to_px(xs[i], ys[i]);
        const cv::Scalar color = iterations[i].guardrail_triggered ? palette::kCritical : palette::kSeries1;
        cv::circle(canvas, p, 5, color, cv::FILLED, cv::LINE_AA);
        cv::putText(canvas, std::to_string(iterations[i].n_inliers), cv::Point(p.x + 8, p.y - 8), cv::FONT_HERSHEY_SIMPLEX,
            0.4, palette::kText, 1, cv::LINE_AA);
    }
    cv::putText(canvas, "gate iteration", cv::Point(plot_rect.x, plot_rect.y + plot_rect.height + 24),
        cv::FONT_HERSHEY_SIMPLEX, 0.45, palette::kText, 1, cv::LINE_AA);
    return canvas;
}

// Funnel table (§6): 7 fixed rows, not worth a chart -- rendered as plain text, but at fixed pixel
// column positions rather than space-padded strings, since HERSHEY_SIMPLEX isn't truly monospace
// and space-padding doesn't actually line up into clean columns under a proportional font.
inline cv::Mat render_funnel_table(const ClutterFunnel& funnel, int width = 520, int height = 220) {
    cv::Mat canvas = make_blank_canvas(width, height);
    constexpr int kCol0 = 10, kCol1 = 300, kCol2 = 410;
    constexpr double kFontScale = 0.42;

    int y = 24;
    cv::putText(canvas, "stage", cv::Point(kCol0, y), cv::FONT_HERSHEY_SIMPLEX, kFontScale, palette::kText, 1, cv::LINE_AA);
    cv::putText(
        canvas, "entering", cv::Point(kCol1, y), cv::FONT_HERSHEY_SIMPLEX, kFontScale, palette::kText, 1, cv::LINE_AA);
    cv::putText(
        canvas, "surviving", cv::Point(kCol2, y), cv::FONT_HERSHEY_SIMPLEX, kFontScale, palette::kText, 1, cv::LINE_AA);
    y += 24;

    for (const auto& stage : funnel) {
        cv::putText(
            canvas, stage.stage_name, cv::Point(kCol0, y), cv::FONT_HERSHEY_SIMPLEX, kFontScale, palette::kText, 1, cv::LINE_AA);
        cv::putText(canvas, std::to_string(stage.n_entering), cv::Point(kCol1, y), cv::FONT_HERSHEY_SIMPLEX, kFontScale,
            palette::kText, 1, cv::LINE_AA);
        cv::putText(canvas, std::to_string(stage.n_surviving), cv::Point(kCol2, y), cv::FONT_HERSHEY_SIMPLEX, kFontScale,
            palette::kText, 1, cv::LINE_AA);
        y += 22;
    }
    return canvas;
}

}  // namespace radarcam
}  // namespace bev
