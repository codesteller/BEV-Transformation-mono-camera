#include <QApplication>
#include <QComboBox>
#include <QDateTime>
#include <QFileDialog>
#include <QFileInfo>
#include <QFormLayout>
#include <QFrame>
#include <QFont>
#include <QGroupBox>
#include <QHBoxLayout>
#include <QImage>
#include <QLabel>
#include <QLineEdit>
#include <QMainWindow>
#include <QPixmap>
#include <QPushButton>
#include <QToolButton>
#include <QSlider>
#include <QSpinBox>
#include <QTabWidget>
#include <QTimer>
#include <QVBoxLayout>
#include <QWidget>

#include <opencv2/calib3d.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio/registry.hpp>
#include <opencv2/videoio.hpp>

#include <filesystem>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <string>
#include <algorithm>
#include <cctype>
#include <array>
#include <cmath>
#include <limits>
#include <vector>

static QImage mat_to_qimage(const cv::Mat& mat) {
    if (mat.empty()) {
        return QImage();
    }
    cv::Mat rgb;
    cv::cvtColor(mat, rgb, cv::COLOR_BGR2RGB);
    return QImage(rgb.data, rgb.cols, rgb.rows, static_cast<int>(rgb.step), QImage::Format_RGB888)
        .copy();
}

static QWidget* make_brand_header() {
    auto* header = new QFrame();
    auto* layout = new QHBoxLayout(header);
    layout->setContentsMargins(12, 10, 12, 10);
    layout->setSpacing(12);

    auto* logo_label = new QLabel();
    logo_label->setFixedSize(72, 72);
    logo_label->setAlignment(Qt::AlignCenter);
    logo_label->setStyleSheet(
        "background:#ffffff; border:1px solid #d4dce7; border-radius:12px; color:#6b7785; font-size:11px;");

    QPixmap logo(":/assets/pi_logo.png");
    if (logo.isNull()) {
        const QStringList fallback_paths = {
            "../assets/pi_logo.png",
            "cpp/assets/pi_logo.png",
            "assets/pi_logo.png"
        };
        for (const QString& path : fallback_paths) {
            if (QFileInfo::exists(path) && logo.load(path)) {
                break;
            }
        }
    }

    if (!logo.isNull()) {
        logo_label->setPixmap(
            logo.scaled(64, 64, Qt::KeepAspectRatio, Qt::SmoothTransformation));
    } else {
        logo_label->setText("Logo\nMissing");
    }

    auto* title = new QLabel("Gahan AI");
    QFont title_font = title->font();
    title_font.setPointSize(18);
    title_font.setBold(true);
    title->setFont(title_font);

    auto* subtitle = new QLabel("Monocular Camera Calibration Tool");
    subtitle->setStyleSheet("color: #5a5a5a;");

    auto* text_block = new QWidget();
    auto* text_layout = new QVBoxLayout(text_block);
    text_layout->setContentsMargins(0, 0, 0, 0);
    text_layout->setSpacing(2);
    text_layout->addWidget(title);
    text_layout->addWidget(subtitle);

    layout->addWidget(logo_label);
    layout->addWidget(text_block);
    layout->addStretch(1);

    return header;
}

static QToolButton* make_toggle_button(const QString& label) {
    auto* button = new QToolButton();
    button->setCheckable(true);
    button->setAutoRaise(false);
    button->setToolButtonStyle(Qt::ToolButtonTextOnly);
    button->setProperty("toggleLabel", label);
    return button;
}

static void set_state_label(QLabel* label, const QString& text, const QString& color_hex) {
    label->setText(QString("<span style='color:%1;font-weight:700;'>%2</span>").arg(color_hex, text));
}

static void set_toggle_button_visual(QToolButton* button, bool enabled_state) {
    const QString label = button->property("toggleLabel").toString();
    button->setText(QString("%1 %2 %3")
        .arg(enabled_state ? QStringLiteral("●") : QStringLiteral("○"),
             label,
             enabled_state ? QStringLiteral("ON") : QStringLiteral("OFF")));
    button->setProperty("toggleState", enabled_state ? "on" : "off");
    button->style()->unpolish(button);
    button->style()->polish(button);
}

class IntrinsicsTab final : public QWidget {
public:
    explicit IntrinsicsTab(QWidget* parent = nullptr) : QWidget(parent) {
        auto* root_layout = new QHBoxLayout(this);
        root_layout->setSpacing(14);
        root_layout->setContentsMargins(10, 10, 10, 10);
        setObjectName("intrinsicsTabRoot");

        auto* left_panel = new QGroupBox("Intrinsics Inputs");
        auto* left_form = new QFormLayout(left_panel);

        camera_combo_ = new QComboBox();
        refresh_cameras_btn_ = new QPushButton("Refresh");
        auto* cam_row = new QWidget();
        auto* cam_row_layout = new QHBoxLayout(cam_row);
        cam_row_layout->setContentsMargins(0, 0, 0, 0);
        cam_row_layout->addWidget(camera_combo_, 1);
        cam_row_layout->addWidget(refresh_cameras_btn_);

        corners_x_ = new QSpinBox();
        corners_x_->setRange(3, 30);
        corners_x_->setValue(10);

        corners_y_ = new QSpinBox();
        corners_y_->setRange(3, 30);
        corners_y_->setValue(7);

        square_size_ = new QLineEdit("0.023");

        save_path_ = new QLineEdit("intrinsics.yaml");
        browse_button_ = new QPushButton("Browse");
        auto* save_row = new QWidget();
        auto* save_row_layout = new QHBoxLayout(save_row);
        save_row_layout->setContentsMargins(0, 0, 0, 0);
        save_row_layout->addWidget(save_path_, 1);
        save_row_layout->addWidget(browse_button_);

        frame_folder_ = new QLineEdit("intrinsics_frames");
        auto* frame_folder_row = new QWidget();
        auto* frame_folder_layout = new QHBoxLayout(frame_folder_row);
        frame_folder_layout->setContentsMargins(0, 0, 0, 0);
        frame_folder_layout->addWidget(frame_folder_, 1);

        left_form->addRow("Camera", cam_row);
        left_form->addRow("Chessboard Inner Corners X", corners_x_);
        left_form->addRow("Chessboard Inner Corners Y", corners_y_);
        left_form->addRow("Square Size (m)", square_size_);
        left_form->addRow("Save YAML", save_row);
        left_form->addRow("Save Frames Folder", frame_folder_row);

        auto* mode_card = new QGroupBox("Capture Controls");
        auto* controls_layout = new QVBoxLayout(mode_card);
        controls_layout->setContentsMargins(12, 14, 12, 12);
        controls_layout->setSpacing(10);

        preview_btn_ = make_toggle_button("Preview");
        preview_btn_->setMinimumHeight(34);
        auto_capture_btn_ = make_toggle_button("Auto Capture");
        auto_capture_btn_->setMinimumHeight(34);
        auto_capture_btn_->setStyleSheet("QToolButton { padding: 8px 12px; }");

        capture_button_ = new QPushButton("Capture Frame");
        calibrate_button_ = new QPushButton("Calibrate + Save");
        clear_button_ = new QPushButton("Clear Captures");

        auto* row1 = new QWidget();
        auto* row1_layout = new QHBoxLayout(row1);
        row1_layout->setContentsMargins(0, 0, 0, 0);
        row1_layout->addWidget(preview_btn_);
        row1_layout->addWidget(auto_capture_btn_);

        auto* row2 = new QWidget();
        auto* row2_layout = new QHBoxLayout(row2);
        row2_layout->setContentsMargins(0, 0, 0, 0);
        row2_layout->addWidget(capture_button_);
        row2_layout->addWidget(calibrate_button_);
        row2_layout->addWidget(clear_button_);

        controls_layout->addWidget(row1);
        controls_layout->addWidget(row2);

        auto_state_label_ = new QLabel();
        auto_state_label_->setTextFormat(Qt::RichText);
        auto_state_label_->setAlignment(Qt::AlignLeft | Qt::AlignVCenter);
        set_state_label(auto_state_label_, "Auto-capture OFF", "#c62828");

        auto* coverage_panel = new QGroupBox("Coverage (ROS-style)");
        auto* coverage_form = new QFormLayout(coverage_panel);
        x_cov_slider_ = create_cov_slider();
        y_cov_slider_ = create_cov_slider();
        size_cov_slider_ = create_cov_slider();
        skew_cov_slider_ = create_cov_slider();
        x_cov_label_ = new QLabel("0%");
        y_cov_label_ = new QLabel("0%");
        size_cov_label_ = new QLabel("0%");
        skew_cov_label_ = new QLabel("0%");
        coverage_form->addRow("X Coverage", make_cov_row(x_cov_slider_, x_cov_label_));
        coverage_form->addRow("Y Coverage", make_cov_row(y_cov_slider_, y_cov_label_));
        coverage_form->addRow("Size/Scale Coverage", make_cov_row(size_cov_slider_, size_cov_label_));
        coverage_form->addRow("Skew Coverage", make_cov_row(skew_cov_slider_, skew_cov_label_));

        captures_label_ = new QLabel("Captured frames: 0");
        status_label_ = new QLabel("Ready.");
        status_label_->setWordWrap(true);

        info_banner_ = new QLabel("Select a camera, start preview, then toggle auto-capture to start automatic frame collection.");
        info_banner_->setWordWrap(true);
        info_banner_->setObjectName("infoBanner");
        info_banner_->setMinimumHeight(38);

        auto* left_stack = new QVBoxLayout();
        left_stack->addWidget(left_panel);
        left_stack->addWidget(mode_card);
        left_stack->addWidget(info_banner_);
        left_stack->addWidget(auto_state_label_);
        left_stack->addWidget(coverage_panel);
        left_stack->addWidget(captures_label_);
        left_stack->addWidget(status_label_);
        left_stack->addStretch(1);

        auto* left_container = new QWidget();
        left_container->setLayout(left_stack);

        auto* right_panel = new QGroupBox("View Finder");
        auto* right_layout = new QVBoxLayout(right_panel);
        preview_label_ = new QLabel("Preview stopped.");
        preview_label_->setAlignment(Qt::AlignCenter);
        preview_label_->setMinimumSize(800, 520);
        preview_label_->setObjectName("previewArea");
        right_layout->addWidget(preview_label_);

        root_layout->addWidget(left_container, 1);
        root_layout->addWidget(right_panel, 3);

        timer_ = new QTimer(this);
        timer_->setInterval(33);

        connect(refresh_cameras_btn_, &QPushButton::clicked, this, [this]() { refresh_cameras(); });
        connect(preview_btn_, &QToolButton::toggled, this, [this](bool checked) {
            if (checked) {
                start_preview();
            } else {
                stop_preview();
            }
        });
        connect(auto_capture_btn_, &QToolButton::toggled, this, [this](bool checked) {
            if (checked) {
                start_auto_capture();
            } else {
                stop_auto_capture();
            }
        });
        connect(capture_button_, &QPushButton::clicked, this, [this]() { capture_frame(); });
        connect(calibrate_button_, &QPushButton::clicked, this, [this]() { calibrate_and_save(); });
        connect(clear_button_, &QPushButton::clicked, this, [this]() { clear_captures(); });
        connect(browse_button_, &QPushButton::clicked, this, [this]() { browse_save_path(); });
        connect(timer_, &QTimer::timeout, this, [this]() { on_frame_tick(); });

        reset_coverage();
        apply_ui_state();
        refresh_cameras();
    }

    ~IntrinsicsTab() override {
        stop_preview();
    }

private:
    static QSlider* create_cov_slider() {
        auto* slider = new QSlider(Qt::Horizontal);
        slider->setRange(0, 100);
        slider->setValue(0);
        slider->setEnabled(false);
        return slider;
    }

    static QWidget* make_cov_row(QSlider* slider, QLabel* value_label) {
        auto* row = new QWidget();
        auto* layout = new QHBoxLayout(row);
        layout->setContentsMargins(0, 0, 0, 0);
        layout->addWidget(slider, 1);
        layout->addWidget(value_label);
        return row;
    }

    static std::array<double, 4> sample_params(
        const std::vector<cv::Point2f>& corners,
        const cv::Size& frame_size,
        int nx,
        int ny) {
        const cv::Point2f up_left = corners.front();
        const cv::Point2f up_right = corners[nx - 1];
        const cv::Point2f down_right = corners.back();
        const cv::Point2f down_left = corners[(ny - 1) * nx];

        const cv::Point2f a = up_right - up_left;
        const cv::Point2f b = down_right - up_right;
        const cv::Point2f c = down_left - down_right;
        const cv::Point2f p = b + c;
        const cv::Point2f q = a + b;
        const double area = std::abs(p.x * q.y - p.y * q.x) / 2.0;

        const cv::Point2f ab = up_left - up_right;
        const cv::Point2f cb = down_right - up_right;
        const double denom = std::sqrt(ab.dot(ab)) * std::sqrt(cb.dot(cb));
        double cos_val = 1.0;
        if (denom > 1e-9) {
            cos_val = std::clamp(ab.dot(cb) / denom, -1.0, 1.0);
        }
        const double angle = std::acos(cos_val);
        const double skew = std::min(1.0, 2.0 * std::abs((CV_PI / 2.0) - angle));

        double mean_x = 0.0;
        double mean_y = 0.0;
        for (const auto& pt : corners) {
            mean_x += pt.x;
            mean_y += pt.y;
        }
        mean_x /= static_cast<double>(corners.size());
        mean_y /= static_cast<double>(corners.size());

        const double width = static_cast<double>(frame_size.width);
        const double height = static_cast<double>(frame_size.height);
        const double border = std::sqrt(std::max(area, 0.0));

        double p_x = 0.5;
        double p_y = 0.5;
        if (width - border > 1.0) {
            p_x = std::clamp((mean_x - border / 2.0) / (width - border), 0.0, 1.0);
        }
        if (height - border > 1.0) {
            p_y = std::clamp((mean_y - border / 2.0) / (height - border), 0.0, 1.0);
        }

        const double p_size = (width > 0.0 && height > 0.0)
            ? std::sqrt(std::max(0.0, area / (width * height)))
            : 0.0;

        return {p_x, p_y, p_size, skew};
    }

    void reset_coverage() {
        coverage_initialized_ = false;
        x_cov_slider_->setValue(0);
        y_cov_slider_->setValue(0);
        size_cov_slider_->setValue(0);
        skew_cov_slider_->setValue(0);
        x_cov_label_->setText("0%");
        y_cov_label_->setText("0%");
        size_cov_label_->setText("0%");
        skew_cov_label_->setText("0%");
    }

    void update_coverage(const std::vector<cv::Point2f>& corners) {
        const std::array<double, 4> params = sample_params(
            corners, last_frame_size_, corners_x_->value(), corners_y_->value());

        if (!coverage_initialized_) {
            coverage_min_ = params;
            coverage_max_ = params;
            coverage_initialized_ = true;
        } else {
            for (int i = 0; i < 4; ++i) {
                coverage_min_[i] = std::min(coverage_min_[i], params[i]);
                coverage_max_[i] = std::max(coverage_max_[i], params[i]);
            }
        }

        std::array<int, 4> progress{};
        for (int i = 0; i < 4; ++i) {
            const double covered = coverage_max_[i] - coverage_min_[i];
            const double ratio = std::clamp(covered / k_target_ranges_[i], 0.0, 1.0);
            progress[i] = static_cast<int>(ratio * 100.0 + 0.5);
        }

        x_cov_slider_->setValue(progress[0]);
        y_cov_slider_->setValue(progress[1]);
        size_cov_slider_->setValue(progress[2]);
        skew_cov_slider_->setValue(progress[3]);
        x_cov_label_->setText(QString("%1%").arg(progress[0]));
        y_cov_label_->setText(QString("%1%").arg(progress[1]));
        size_cov_label_->setText(QString("%1%").arg(progress[2]));
        skew_cov_label_->setText(QString("%1%").arg(progress[3]));
    }

    void start_auto_capture() {
        if (!cap_.isOpened()) {
            status_label_->setText("Start preview before starting calibration.");
            auto_capture_btn_->blockSignals(true);
            auto_capture_btn_->setChecked(false);
            auto_capture_btn_->blockSignals(false);
            set_toggle_button_visual(auto_capture_btn_, false);
            return;
        }
        auto_capture_enabled_ = true;
        force_auto_capture_once_ = true;
        set_state_label(auto_state_label_, "Auto-capture ON", "#2e7d32");
        info_banner_->setText("Auto-capture is active. Captured frames will be saved with the calibration output.");
        status_label_->setText("Calibration mode started. Valid detections are auto-captured.");
        set_toggle_button_visual(auto_capture_btn_, true);
        apply_ui_state();
    }

    void stop_auto_capture() {
        auto_capture_enabled_ = false;
        set_state_label(auto_state_label_, "Auto-capture OFF", "#c62828");
        info_banner_->setText("Auto-capture is off. Use Capture Frame for manual captures or toggle Auto Capture to resume.");
        set_toggle_button_visual(auto_capture_btn_, false);
        apply_ui_state();
    }

    bool should_auto_capture(const std::vector<cv::Point2f>& corners) {
        const std::array<double, 4> params = sample_params(
            corners, last_frame_size_, corners_x_->value(), corners_y_->value());

        if (force_auto_capture_once_) {
            force_auto_capture_once_ = false;
            return true;
        }

        const qint64 now_ms = QDateTime::currentMSecsSinceEpoch();
        if (now_ms - last_auto_capture_ms_ < 300) {
            return false;
        }

        if (!captured_params_.empty()) {
            double best_dist = std::numeric_limits<double>::max();
            for (const auto& prev : captured_params_) {
                double d2 = 0.0;
                for (int i = 0; i < 4; ++i) {
                    const double s = std::max(k_target_ranges_[i], 1e-6);
                    const double delta = (params[i] - prev[i]) / s;
                    d2 += delta * delta;
                }
                best_dist = std::min(best_dist, std::sqrt(d2));
            }

            // Accept only if the new observation is sufficiently different in coverage space.
            if (best_dist < 0.18) {
                return false;
            }
        }

        last_auto_capture_ms_ = now_ms;
        return true;
    }

    static std::vector<int> enumerate_video_indices() {
        std::vector<int> indices;
        const std::filesystem::path dev_path("/dev");
        if (!std::filesystem::exists(dev_path)) {
            return indices;
        }

        for (const auto& entry : std::filesystem::directory_iterator(dev_path)) {
            const std::string name = entry.path().filename().string();
            if (name.rfind("video", 0) != 0) {
                continue;
            }
            const std::string suffix = name.substr(5);
            if (suffix.empty() || !std::all_of(suffix.begin(), suffix.end(), [](unsigned char c) {
                    return std::isdigit(c) != 0;
                })) {
                continue;
            }
            indices.push_back(std::stoi(suffix));
        }

        std::sort(indices.begin(), indices.end());
        return indices;
    }

    void refresh_cameras() {
        camera_combo_->clear();
        const std::vector<int> candidates = enumerate_video_indices();
        for (const int idx : candidates) {
            cv::VideoCapture probe(idx, cv::CAP_V4L2);
            if (probe.isOpened()) {
                camera_combo_->addItem(QString("Camera %1").arg(idx), idx);
                probe.release();
            }
        }
        if (camera_combo_->count() == 0) {
            status_label_->setText("No cameras found. Connect a camera and press Refresh.");
        } else {
            status_label_->setText(QString("Found %1 camera(s).").arg(camera_combo_->count()));
        }
    }

    void start_preview() {
        if (camera_combo_->count() == 0) {
            status_label_->setText("Cannot start preview: no camera selected.");
            return;
        }

        const int device_idx = camera_combo_->currentData().toInt();
        cap_.open(device_idx, cv::CAP_V4L2);
        if (!cap_.isOpened()) {
            cap_.open(device_idx, cv::CAP_ANY);
        }
        if (!cap_.isOpened()) {
            status_label_->setText(QString("Failed to open camera %1.").arg(device_idx));
            return;
        }

        preview_btn_->blockSignals(true);
        preview_btn_->setChecked(true);
        preview_btn_->blockSignals(false);
        set_toggle_button_visual(preview_btn_, true);

        timer_->start();
        status_label_->setText(QString("Preview started on camera %1.").arg(device_idx));
        apply_ui_state();
    }

    void stop_preview() {
        stop_auto_capture();
        if (timer_->isActive()) {
            timer_->stop();
        }
        if (cap_.isOpened()) {
            cap_.release();
        }
        preview_btn_->blockSignals(true);
        preview_btn_->setChecked(false);
        preview_btn_->setText("Preview Off");
        preview_btn_->blockSignals(false);
        set_toggle_button_visual(preview_btn_, false);
        preview_label_->setText("Preview stopped.");
        apply_ui_state();
    }

    void on_frame_tick() {
        if (!cap_.isOpened()) {
            return;
        }

        cv::Mat frame;
        if (!cap_.read(frame) || frame.empty()) {
            status_label_->setText("Camera stream ended or frame read failed.");
            return;
        }

        last_frame_clone_ = frame.clone();

        cv::Mat overlay = frame.clone();
        std::vector<cv::Point2f> corners;
        const cv::Size board(corners_x_->value(), corners_y_->value());
        const bool found = cv::findChessboardCorners(
            overlay, board, corners,
            cv::CALIB_CB_ADAPTIVE_THRESH | cv::CALIB_CB_NORMALIZE_IMAGE);

        if (found) {
            cv::Mat gray;
            cv::cvtColor(overlay, gray, cv::COLOR_BGR2GRAY);
            cv::cornerSubPix(
                gray,
                corners,
                cv::Size(11, 11),
                cv::Size(-1, -1),
                cv::TermCriteria(cv::TermCriteria::EPS | cv::TermCriteria::MAX_ITER, 30, 0.1));
            cv::drawChessboardCorners(overlay, board, corners, found);
            last_detected_corners_ = corners;
            last_detection_valid_ = true;
            update_coverage(corners);

            if (auto_capture_enabled_ && should_auto_capture(corners)) {
                image_points_.push_back(corners);
                captured_params_.push_back(sample_params(
                    corners, last_frame_size_, corners_x_->value(), corners_y_->value()));
                captured_frames_.push_back(frame.clone());
                captures_label_->setText(QString("Captured frames: %1").arg(image_points_.size()));
                status_label_->setText(QString("Auto-captured frame %1.").arg(image_points_.size()));
            }
        } else {
            last_detection_valid_ = false;
        }

        last_frame_size_ = frame.size();
        const QImage image = mat_to_qimage(overlay);
        if (!image.isNull()) {
            preview_label_->setPixmap(QPixmap::fromImage(image).scaled(
                preview_label_->size(), Qt::KeepAspectRatio, Qt::SmoothTransformation));
        }
    }

    void capture_frame() {
        if (!cap_.isOpened()) {
            status_label_->setText("Start preview before capturing.");
            return;
        }
        if (!last_detection_valid_) {
            status_label_->setText("No chessboard detected in current frame.");
            return;
        }

        image_points_.push_back(last_detected_corners_);
        captured_params_.push_back(sample_params(
            last_detected_corners_, last_frame_size_, corners_x_->value(), corners_y_->value()));
        if (!last_frame_clone_.empty()) {
            captured_frames_.push_back(last_frame_clone_.clone());
        }
        captures_label_->setText(QString("Captured frames: %1").arg(image_points_.size()));
        status_label_->setText("Capture accepted.");
    }

    void clear_captures() {
        image_points_.clear();
        captured_params_.clear();
        captured_frames_.clear();
        last_auto_capture_ms_ = 0;
        force_auto_capture_once_ = false;
        reset_coverage();
        captures_label_->setText("Captured frames: 0");
        status_label_->setText("Captured set cleared.");
    }

    void browse_save_path() {
        const QString path = QFileDialog::getSaveFileName(
            this,
            "Save Intrinsics YAML",
            save_path_->text(),
            "YAML files (*.yaml *.yml)");
        if (!path.isEmpty()) {
            save_path_->setText(path);
        }
    }

    void calibrate_and_save() {
        if (image_points_.size() < 5) {
            status_label_->setText("Capture at least 5 valid frames before calibration.");
            return;
        }
        if (last_frame_size_.width <= 0 || last_frame_size_.height <= 0) {
            status_label_->setText("No valid frame size from preview.");
            return;
        }

        bool ok = false;
        const double square = square_size_->text().toDouble(&ok);
        if (!ok || square <= 0.0) {
            status_label_->setText("Square size must be a positive number.");
            return;
        }

        const int nx = corners_x_->value();
        const int ny = corners_y_->value();
        std::vector<cv::Point3f> object_template;
        object_template.reserve(nx * ny);
        for (int y = 0; y < ny; ++y) {
            for (int x = 0; x < nx; ++x) {
                object_template.emplace_back(static_cast<float>(x * square), static_cast<float>(y * square), 0.0f);
            }
        }

        std::vector<std::vector<cv::Point3f>> object_points(image_points_.size(), object_template);
        cv::Mat camera_matrix = cv::Mat::eye(3, 3, CV_64F);
        cv::Mat dist_coeffs = cv::Mat::zeros(8, 1, CV_64F);
        std::vector<cv::Mat> rvecs;
        std::vector<cv::Mat> tvecs;

        const double rms = cv::calibrateCamera(
            object_points,
            image_points_,
            last_frame_size_,
            camera_matrix,
            dist_coeffs,
            rvecs,
            tvecs);

        // Match the YAML schema used by python/assets/BR01FU9650-1920x1020.yaml.
        cv::Mat rectification = cv::Mat::eye(3, 3, CV_64F);
        cv::Mat ncm = cv::getOptimalNewCameraMatrix(
            camera_matrix,
            dist_coeffs,
            last_frame_size_,
            0.0,
            last_frame_size_);

        cv::Mat projection = cv::Mat::zeros(3, 4, CV_64F);
        for (int r = 0; r < 3; ++r) {
            for (int c = 0; c < 3; ++c) {
                projection.at<double>(r, c) = ncm.at<double>(r, c);
            }
        }

        std::array<double, 5> d5{};
        for (int i = 0; i < 5; ++i) {
            const int idx = std::min(i, static_cast<int>(dist_coeffs.total()) - 1);
            d5[i] = dist_coeffs.at<double>(idx, 0);
        }

        const std::string path = save_path_->text().toStdString();
        try {
            const std::filesystem::path out_path(path);
            if (out_path.has_parent_path() && !out_path.parent_path().empty()) {
                std::filesystem::create_directories(out_path.parent_path());
            }
        } catch (const std::exception&) {
            status_label_->setText("Invalid output path.");
            return;
        }

        std::filesystem::path yaml_path(path);
        std::filesystem::path image_dir = yaml_path.parent_path();
        if (image_dir.empty()) {
            image_dir = std::filesystem::current_path();
        }
        image_dir /= yaml_path.stem();
        image_dir += "_images";
        std::filesystem::create_directories(image_dir);

        for (size_t i = 0; i < captured_frames_.size(); ++i) {
            const std::filesystem::path image_path = image_dir / ("frame_" + std::to_string(i) + ".png");
            cv::imwrite(image_path.string(), captured_frames_[i]);
        }

          std::ofstream fs(path, std::ios::out | std::ios::trunc);
          if (!fs.is_open()) {
            status_label_->setText("Failed to open output path for YAML write.");
            return;
        }

          fs << std::fixed;
          fs << "image_width: " << last_frame_size_.width << "\n";
          fs << "image_height: " << last_frame_size_.height << "\n";
          fs << "camera_name: PINHOLE\n";
          fs << "camera_matrix:\n";
          fs << "  rows: 3\n";
          fs << "  cols: 3\n";
          fs << "  data: [";
          fs << std::setprecision(5)
              << camera_matrix.at<double>(0, 0) << ", " << camera_matrix.at<double>(0, 1) << ", " << camera_matrix.at<double>(0, 2) << ",\n"
              << "            "
              << camera_matrix.at<double>(1, 0) << ", " << camera_matrix.at<double>(1, 1) << ", " << camera_matrix.at<double>(1, 2) << ",\n"
              << "            "
              << camera_matrix.at<double>(2, 0) << ", " << camera_matrix.at<double>(2, 1) << ", " << camera_matrix.at<double>(2, 2) << "]\n";

          fs << "distortion_model: plumb_bob\n";
          fs << "distortion_coefficients:\n";
          fs << "  rows: 1\n";
          fs << "  cols: 5\n";
          fs << "  data: [" << std::setprecision(6)
              << d5[0] << ", " << d5[1] << ", " << d5[2] << ", " << d5[3] << ", " << d5[4] << "]\n";

          fs << "rectification_matrix:\n";
          fs << "  rows: 3\n";
          fs << "  cols: 3\n";
          fs << "  data: [" << std::setprecision(1)
              << rectification.at<double>(0, 0) << ", " << rectification.at<double>(0, 1) << ", " << rectification.at<double>(0, 2) << ",\n"
              << "         "
              << rectification.at<double>(1, 0) << ", " << rectification.at<double>(1, 1) << ", " << rectification.at<double>(1, 2) << ",\n"
              << "         "
              << rectification.at<double>(2, 0) << ", " << rectification.at<double>(2, 1) << ", " << rectification.at<double>(2, 2) << "]\n";

          fs << "projection_matrix:\n";
          fs << "  rows: 3\n";
          fs << "  cols: 4\n";
          fs << "  data: [" << std::setprecision(5)
              << projection.at<double>(0, 0) << ", " << projection.at<double>(0, 1) << ", " << projection.at<double>(0, 2) << ", " << projection.at<double>(0, 3) << ",\n"
              << "            "
              << projection.at<double>(1, 0) << ", " << projection.at<double>(1, 1) << ", " << projection.at<double>(1, 2) << ", " << projection.at<double>(1, 3) << ",\n"
              << "            "
              << projection.at<double>(2, 0) << ", " << projection.at<double>(2, 1) << ", " << projection.at<double>(2, 2) << ", " << projection.at<double>(2, 3) << "]\n";

        fs.close();

        std::ostringstream oss;
        oss << "Calibration complete. RMS error=" << rms << ". Saved: " << path
            << ", images: " << image_dir.string();
        status_label_->setText(QString::fromStdString(oss.str()));
    }

    void apply_ui_state() {
        const bool preview_on = cap_.isOpened();
        const bool auto_on = auto_capture_enabled_;

        camera_combo_->setEnabled(!preview_on);
        refresh_cameras_btn_->setEnabled(!preview_on);
        set_toggle_button_visual(preview_btn_, preview_on);
        set_toggle_button_visual(auto_capture_btn_, auto_on);
        if (preview_on) {
            info_banner_->setText("Preview is running. Camera selection is locked until you stop preview.");
        }

        const bool can_calibrate = !auto_on && !image_points_.empty();
        // Disable actions that should not be used while preview or auto-capture is active.
        calibrate_button_->setEnabled(can_calibrate);
        clear_button_->setEnabled(!auto_on);
        capture_button_->setEnabled(preview_on && !auto_on);
        browse_button_->setEnabled(!preview_on && !auto_on);
        corners_x_->setEnabled(!preview_on);
        corners_y_->setEnabled(!preview_on);
        square_size_->setEnabled(!preview_on);
        save_path_->setEnabled(!preview_on);
        frame_folder_->setEnabled(!preview_on);
    }

private:
    QComboBox* camera_combo_ = nullptr;
    QPushButton* refresh_cameras_btn_ = nullptr;
    QSpinBox* corners_x_ = nullptr;
    QSpinBox* corners_y_ = nullptr;
    QLineEdit* square_size_ = nullptr;
    QLineEdit* save_path_ = nullptr;
    QLineEdit* frame_folder_ = nullptr;
    QLabel* preview_label_ = nullptr;
    QLabel* captures_label_ = nullptr;
    QLabel* status_label_ = nullptr;
    QLabel* auto_state_label_ = nullptr;
    QLabel* info_banner_ = nullptr;
    QSlider* x_cov_slider_ = nullptr;
    QSlider* y_cov_slider_ = nullptr;
    QSlider* size_cov_slider_ = nullptr;
    QSlider* skew_cov_slider_ = nullptr;
    QLabel* x_cov_label_ = nullptr;
    QLabel* y_cov_label_ = nullptr;
    QLabel* size_cov_label_ = nullptr;
    QLabel* skew_cov_label_ = nullptr;
    QTimer* timer_ = nullptr;
    QToolButton* preview_btn_ = nullptr;
    QToolButton* auto_capture_btn_ = nullptr;
    QPushButton* calibrate_button_ = nullptr;
    QPushButton* clear_button_ = nullptr;
    QPushButton* capture_button_ = nullptr;
    QPushButton* browse_button_ = nullptr;

    cv::VideoCapture cap_;
    cv::Size last_frame_size_;
    cv::Mat last_frame_clone_;
    std::vector<cv::Point2f> last_detected_corners_;
    bool last_detection_valid_ = false;
    std::vector<std::vector<cv::Point2f>> image_points_;
    std::vector<cv::Mat> captured_frames_;
    bool auto_capture_enabled_ = false;
    qint64 last_auto_capture_ms_ = 0;
    bool force_auto_capture_once_ = false;
    std::vector<std::array<double, 4>> captured_params_;
    bool coverage_initialized_ = false;
    std::array<double, 4> coverage_min_{};
    std::array<double, 4> coverage_max_{};
    const std::array<double, 4> k_target_ranges_ = {0.7, 0.7, 0.4, 0.5};
};

static QWidget* make_homography_tab() {
    auto* tab = new QWidget();
    auto* root_layout = new QHBoxLayout(tab);

    auto* left_panel = new QGroupBox("Homography Inputs");
    auto* left_form = new QFormLayout(left_panel);
    left_form->addRow("Marker Grid Width (m)", new QLineEdit("2.0"));
    left_form->addRow("Marker Grid Height (m)", new QLineEdit("2.0"));
    left_form->addRow("Camera to Ground (m)", new QLineEdit("1.5"));

    auto* right_panel = new QGroupBox("View Finder");
    auto* right_layout = new QVBoxLayout(right_panel);
    right_layout->addWidget(
        new QLabel("Live camera feed with ArUco ID overlays will be rendered here."));

    root_layout->addWidget(left_panel, 1);
    root_layout->addWidget(right_panel, 3);
    return tab;
}

int main(int argc, char** argv) {
    QApplication app(argc, argv);

    app.setStyleSheet(R"(
        QWidget#intrinsicsTabRoot {
            background: #eef2f8;
        }
        QMainWindow {
            background: #eef2f8;
        }
        QGroupBox {
            background: #ffffff;
            border: 1px solid #d8e0eb;
            border-radius: 14px;
            margin-top: 18px;
            padding-top: 10px;
            font-weight: 600;
        }
        QGroupBox::title {
            subcontrol-origin: margin;
            left: 12px;
            padding: 0 6px;
            color: #19324a;
        }
        QLabel {
            color: #163049;
        }
        QLabel#infoBanner {
            background: #f7f9fc;
            border: 1px solid #d8e0eb;
            border-radius: 12px;
            padding: 10px 12px;
            color: #26415c;
        }
        QLabel#previewArea {
            background: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #102030, stop:1 #1d2f45);
            border: 1px solid #20354d;
            border-radius: 14px;
            color: #d9e6f2;
        }
        QLineEdit, QComboBox, QSpinBox {
            background: #ffffff;
            border: 1px solid #c8d4e1;
            border-radius: 10px;
            padding: 8px 10px;
            min-height: 26px;
        }
        QLineEdit:disabled, QComboBox:disabled, QSpinBox:disabled {
            background: #eef2f8;
            color: #8693a5;
        }
        QPushButton, QToolButton {
            background: #ffffff;
            border: 1px solid #bfcddb;
            border-radius: 10px;
            padding: 8px 12px;
            min-height: 34px;
            color: #163049;
        }
        QPushButton:hover, QToolButton:hover {
            background: #f5f8fb;
        }
        QPushButton:disabled, QToolButton:disabled {
            background: #edf1f5;
            color: #a0a9b6;
            border-color: #d8e0eb;
        }
        QToolButton[toggleState="on"] {
            background: #e8f5e9;
            border-color: #9ccc65;
            color: #1b5e20;
            font-weight: 700;
        }
        QToolButton[toggleState="off"] {
            background: #ffebee;
            border-color: #ef9a9a;
            color: #b71c1c;
            font-weight: 700;
        }
        QSlider::groove:horizontal {
            border: 1px solid #d5deea;
            height: 8px;
            background: #edf2f7;
            border-radius: 4px;
        }
        QSlider::handle:horizontal {
            background: #2c7be5;
            width: 18px;
            margin: -6px 0;
            border-radius: 9px;
        }
    )");

    QMainWindow window;
    window.setWindowTitle("Calibration Tool (Intrinsics + Homography)");

    auto* tabs = new QTabWidget();
    tabs->addTab(new IntrinsicsTab(), "Intrinsics");
    tabs->addTab(make_homography_tab(), "Homography");

    auto* central = new QWidget();
    auto* root_layout = new QVBoxLayout(central);
    root_layout->setContentsMargins(6, 6, 6, 6);
    root_layout->setSpacing(6);
    root_layout->addWidget(make_brand_header());
    root_layout->addWidget(tabs, 1);

    window.setCentralWidget(central);
    window.resize(1280, 760);
    window.show();

    return app.exec();
}
