#include "bev/calibration_io.hpp"

#include <opencv2/calib3d.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>

namespace {

struct CliOptions {
    int camera_index = 0;
    std::string intrinsics_path;
    std::string homography_path;
    double margin_m = 2.0;   // extra world-space margin shown around the calibrated plane
    double px_per_m = 100.0; // BEV canvas resolution
};

void print_usage(const char* prog) {
    std::cout << "Usage: " << prog << " [camera_index] [--intrinsics PATH] [--homography PATH] "
                                       "[--margin-m N] [--px-per-m N]\n"
              << "  camera_index defaults to 0.\n"
              << "  --intrinsics/--homography default to\n"
              << "  ${HOME}/.calibration/openadas/cam<camera_index>/{intrinsics,homography}.yaml\n"
              << "  --margin-m: extra world-space margin (meters) shown around the calibrated\n"
              << "              plane in the BEV view (default 2.0).\n"
              << "  --px-per-m: BEV canvas resolution in pixels per meter (default 100).\n";
}

// Returns false (and leaves opts untouched further) on --help or a bad argument.
bool parse_args(int argc, char** argv, CliOptions& opts) {
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        auto next_value = [&](const char* flag) -> std::string {
            if (i + 1 >= argc) {
                throw std::runtime_error(std::string("missing value for ") + flag);
            }
            return argv[++i];
        };

        if (arg == "-h" || arg == "--help") {
            return false;
        } else if (arg == "--intrinsics") {
            opts.intrinsics_path = next_value("--intrinsics");
        } else if (arg == "--homography") {
            opts.homography_path = next_value("--homography");
        } else if (arg == "--margin-m") {
            opts.margin_m = std::stod(next_value("--margin-m"));
        } else if (arg == "--px-per-m") {
            opts.px_per_m = std::stod(next_value("--px-per-m"));
        } else if (!arg.empty() && arg[0] != '-') {
            opts.camera_index = std::stoi(arg);
        } else {
            std::cerr << "Unknown argument: " << arg << "\n";
            return false;
        }
    }
    return true;
}

// Draws a 1m grid, the calibrated plane's own footprint, and the camera's ground position onto
// the BEV canvas -- a raw warp with no scale/orientation reference is hard to read at a glance.
// (world_x_min, world_y_min) is the world coordinate mapped to pixel (0,0), matching whatever
// offset the caller baked into its world-to-pixel transform.
void draw_bev_reference(cv::Mat& bev, const bev::HomographyData& homography,
    double world_x_min, double world_y_min, double px_per_m) {
    auto to_px = [&](double wx, double wy) {
        return cv::Point(
            static_cast<int>(std::lround((wx - world_x_min) * px_per_m)),
            static_cast<int>(std::lround((wy - world_y_min) * px_per_m)));
    };
    const double world_x_max = world_x_min + bev.cols / px_per_m;
    const double world_y_max = world_y_min + bev.rows / px_per_m;

    const cv::Scalar grid_color(60, 60, 60);
    for (int gx = static_cast<int>(std::floor(world_x_min)); gx <= static_cast<int>(std::ceil(world_x_max)); ++gx) {
        const int x = to_px(gx, 0).x;
        cv::line(bev, cv::Point(x, 0), cv::Point(x, bev.rows), grid_color, 1);
    }
    for (int gy = static_cast<int>(std::floor(world_y_min)); gy <= static_cast<int>(std::ceil(world_y_max)); ++gy) {
        const int y = to_px(0, gy).y;
        cv::line(bev, cv::Point(0, y), cv::Point(bev.cols, y), grid_color, 1);
    }

    cv::rectangle(bev, to_px(0.0, 0.0), to_px(homography.plane_width_m, homography.plane_height_m),
        cv::Scalar(40, 200, 255), 2);

    const cv::Point camera_px = to_px(homography.camera_ground_xy.x, homography.camera_ground_xy.y);
    if (camera_px.x >= 0 && camera_px.x < bev.cols && camera_px.y >= 0 && camera_px.y < bev.rows) {
        cv::drawMarker(bev, camera_px, cv::Scalar(0, 0, 255), cv::MARKER_TRIANGLE_UP, 16, 2);
    }
}

}  // namespace

int main(int argc, char** argv) {
    CliOptions opts;
    try {
        if (!parse_args(argc, argv, opts)) {
            print_usage(argv[0]);
            return 0;
        }
    } catch (const std::exception& e) {
        std::cerr << "Argument error: " << e.what() << "\n";
        print_usage(argv[0]);
        return 1;
    }

    if (opts.intrinsics_path.empty()) {
        opts.intrinsics_path = bev::default_camera_asset_dir(opts.camera_index) + "/intrinsics.yaml";
    }
    if (opts.homography_path.empty()) {
        opts.homography_path = bev::default_camera_asset_dir(opts.camera_index) + "/homography.yaml";
    }

    bev::IntrinsicsData intrinsics;
    std::string error;
    if (!bev::load_intrinsics_yaml(opts.intrinsics_path, intrinsics, error)) {
        std::cerr << "Failed to load intrinsics from " << opts.intrinsics_path << ": " << error << "\n";
        return 1;
    }

    bev::HomographyData homography;
    if (!bev::load_homography_yaml(opts.homography_path, homography, error)) {
        std::cerr << "Failed to load homography from " << opts.homography_path << ": " << error << "\n";
        return 1;
    }

    if (intrinsics.image_width != homography.image_width || intrinsics.image_height != homography.image_height) {
        std::cerr << "Warning: intrinsics resolution (" << intrinsics.image_width << "x" << intrinsics.image_height
                   << ") does not match the resolution the homography was solved at ("
                   << homography.image_width << "x" << homography.image_height
                   << "). Rectification/warp will be inaccurate -- recalibrate at a consistent resolution.\n";
    }

    cv::VideoCapture cap(opts.camera_index, cv::CAP_V4L2);
    if (!cap.isOpened()) {
        cap.open(opts.camera_index, cv::CAP_ANY);
    }
    if (!cap.isOpened()) {
        std::cerr << "Failed to open camera " << opts.camera_index << ".\n";
        return 1;
    }
    // Same MJPG-before-resolution fix as the calibration tool -- most UVC cameras cap raw YUYV
    // capture well below their real resolution, so FOURCC has to be negotiated first.
    cap.set(cv::CAP_PROP_FOURCC, cv::VideoWriter::fourcc('M', 'J', 'P', 'G'));
    cap.set(cv::CAP_PROP_FRAME_WIDTH, intrinsics.image_width);
    cap.set(cv::CAP_PROP_FRAME_HEIGHT, intrinsics.image_height);

    const int actual_w = static_cast<int>(std::lround(cap.get(cv::CAP_PROP_FRAME_WIDTH)));
    const int actual_h = static_cast<int>(std::lround(cap.get(cv::CAP_PROP_FRAME_HEIGHT)));
    if (actual_w != intrinsics.image_width || actual_h != intrinsics.image_height) {
        std::cerr << "Warning: camera gave " << actual_w << "x" << actual_h
                   << " but intrinsics were calibrated at "
                   << intrinsics.image_width << "x" << intrinsics.image_height << ".\n";
    }

    // homography_matrix (from HomographyTab::solve_and_save) maps rectified-image pixels to
    // world meters. Composing it with a meters-to-pixels similarity transform gives a single
    // warp straight to a BEV canvas. The view window is the calibrated plane's rectangle plus
    // the camera's own ground position (it can sit well outside the plane, e.g. several meters
    // behind its near edge), each padded by --margin-m, so the camera marker is never clipped.
    const double world_x_min = std::min(0.0, static_cast<double>(homography.camera_ground_xy.x)) - opts.margin_m;
    const double world_x_max =
        std::max(homography.plane_width_m, static_cast<double>(homography.camera_ground_xy.x)) + opts.margin_m;
    const double world_y_min = std::min(0.0, static_cast<double>(homography.camera_ground_xy.y)) - opts.margin_m;
    const double world_y_max =
        std::max(homography.plane_height_m, static_cast<double>(homography.camera_ground_xy.y)) + opts.margin_m;

    const int out_w = std::clamp(static_cast<int>(std::lround((world_x_max - world_x_min) * opts.px_per_m)), 1, 4000);
    const int out_h = std::clamp(static_cast<int>(std::lround((world_y_max - world_y_min) * opts.px_per_m)), 1, 4000);

    const cv::Mat world_to_px = (cv::Mat_<double>(3, 3) <<
        opts.px_per_m, 0.0, -world_x_min * opts.px_per_m,
        0.0, opts.px_per_m, -world_y_min * opts.px_per_m,
        0.0, 0.0, 1.0);
    const cv::Mat bev_homography = world_to_px * homography.homography_matrix;

    std::cout << "bev_runner: camera " << opts.camera_index << " at " << actual_w << "x" << actual_h
              << "\n  intrinsics: " << opts.intrinsics_path
              << "\n  homography: " << opts.homography_path
              << "\n  BEV canvas: " << out_w << "x" << out_h << " px (" << opts.px_per_m << " px/m, "
              << opts.margin_m << " m margin)"
              << "\nPress q or Esc to quit.\n";

    cv::namedWindow("Rectified", cv::WINDOW_NORMAL);
    cv::namedWindow("BEV", cv::WINDOW_NORMAL);

    cv::Mat frame;
    cv::Mat rectified;
    cv::Mat bev;
    while (true) {
        if (!cap.read(frame) || frame.empty()) {
            std::cerr << "Frame read failed.\n";
            break;
        }

        cv::undistort(frame, rectified, intrinsics.camera_matrix, intrinsics.dist_coeffs,
            intrinsics.projection_matrix);
        cv::warpPerspective(rectified, bev, bev_homography, cv::Size(out_w, out_h));
        draw_bev_reference(bev, homography, world_x_min, world_y_min, opts.px_per_m);

        cv::imshow("Rectified", rectified);
        cv::imshow("BEV", bev);

        const int key = cv::waitKey(1) & 0xFF;
        if (key == 'q' || key == 27) {
            break;
        }
    }

    return 0;
}
