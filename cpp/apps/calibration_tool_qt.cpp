#include <QApplication>
#include <QComboBox>
#include <QDateTime>
#include <QFileDialog>
#include <QFileInfo>
#include <QFormLayout>
#include <QFrame>
#include <QFont>
#include <QGridLayout>
#include <QGroupBox>
#include <QHBoxLayout>
#include <QImage>
#include <QLabel>
#include <QLineEdit>
#include <QMainWindow>
#include <QMouseEvent>
#include <QPixmap>
#include <QPushButton>
#include <QToolButton>
#include <QScrollArea>
#include <QSlider>
#include <QSpinBox>
#include <QTabWidget>
#include <QTimer>
#include <QVBoxLayout>
#include <QWidget>

#include <opencv2/calib3d.hpp>
#include <opencv2/aruco.hpp>
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
#include <functional>
#include <limits>
#include <set>
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

struct IntrinsicsData {
    int image_width = 0;
    int image_height = 0;
    cv::Mat camera_matrix;      // 3x3 CV_64F (K)
    cv::Mat dist_coeffs;        // Nx1 CV_64F (D)
    cv::Mat projection_matrix;  // 3x3 CV_64F, optional (top-left of ROS-style 3x4 P)
};

// Reads the "key:\n  ...\n  data: [v0, v1, ...]" block written by
// IntrinsicsTab::calibrate_and_save(). Not a general YAML parser -- tailored to that exact
// hand-written schema, searching from block_pos for the next "data: [ ... ]" list.
static std::vector<double> extract_yaml_data_list(const std::string& text, size_t block_pos) {
    std::vector<double> values;
    if (block_pos == std::string::npos) {
        return values;
    }
    const size_t data_pos = text.find("data:", block_pos);
    if (data_pos == std::string::npos) {
        return values;
    }
    const size_t open = text.find('[', data_pos);
    const size_t close = (open == std::string::npos) ? std::string::npos : text.find(']', open);
    if (open == std::string::npos || close == std::string::npos) {
        return values;
    }
    std::stringstream ss(text.substr(open + 1, close - open - 1));
    std::string token;
    while (std::getline(ss, token, ',')) {
        try {
            values.push_back(std::stod(token));
        } catch (const std::exception&) {
        }
    }
    return values;
}

static bool load_intrinsics_yaml(const std::string& path, IntrinsicsData& out, QString& error) {
    std::ifstream fs(path);
    if (!fs.is_open()) {
        error = "Could not open file.";
        return false;
    }
    std::ostringstream buffer;
    buffer << fs.rdbuf();
    const std::string text = buffer.str();

    auto parse_int_after = [&](const std::string& key) -> int {
        const size_t pos = text.find(key);
        if (pos == std::string::npos) {
            return -1;
        }
        try {
            return std::stoi(text.substr(pos + key.size()));
        } catch (const std::exception&) {
            return -1;
        }
    };

    out.image_width = parse_int_after("image_width:");
    out.image_height = parse_int_after("image_height:");
    if (out.image_width <= 0 || out.image_height <= 0) {
        error = "Missing or invalid image_width/image_height.";
        return false;
    }

    const auto k_values = extract_yaml_data_list(text, text.find("camera_matrix:"));
    if (k_values.size() != 9) {
        error = "camera_matrix must have 9 values.";
        return false;
    }
    out.camera_matrix = cv::Mat(3, 3, CV_64F);
    for (int i = 0; i < 9; ++i) {
        out.camera_matrix.at<double>(i / 3, i % 3) = k_values[static_cast<size_t>(i)];
    }

    const auto d_values = extract_yaml_data_list(text, text.find("distortion_coefficients:"));
    if (d_values.empty()) {
        error = "distortion_coefficients missing.";
        return false;
    }
    out.dist_coeffs = cv::Mat(static_cast<int>(d_values.size()), 1, CV_64F);
    for (size_t i = 0; i < d_values.size(); ++i) {
        out.dist_coeffs.at<double>(static_cast<int>(i), 0) = d_values[i];
    }

    const auto p_values = extract_yaml_data_list(text, text.find("projection_matrix:"));
    if (p_values.size() == 12) {
        out.projection_matrix = cv::Mat(3, 3, CV_64F);
        for (int r = 0; r < 3; ++r) {
            for (int c = 0; c < 3; ++c) {
                out.projection_matrix.at<double>(r, c) = p_values[static_cast<size_t>(r * 4 + c)];
            }
        }
    }

    return true;
}

static QWidget* make_brand_header() {
    auto* header = new QFrame();
    header->setObjectName("brandHeader");
    auto* layout = new QHBoxLayout(header);
    layout->setContentsMargins(16, 12, 16, 12);
    layout->setSpacing(14);

    auto* logo_label = new QLabel();
    logo_label->setFixedSize(64, 64);
    logo_label->setAlignment(Qt::AlignCenter);
    logo_label->setStyleSheet(
        "background:#ffffff; border:1px solid #3c3c3c; border-radius:10px; color:#6b7785; font-size:11px;");

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
            logo.scaled(56, 56, Qt::KeepAspectRatio, Qt::SmoothTransformation));
    } else {
        logo_label->setText("Logo\nMissing");
    }

    auto* title = new QLabel("Gahan AI");
    QFont title_font = title->font();
    title_font.setPointSize(15);
    title_font.setBold(true);
    title->setFont(title_font);
    title->setStyleSheet("color:#ffffff;");

    auto* subtitle = new QLabel("Monocular Camera Calibration Tool");
    subtitle->setStyleSheet("color:#9da5b4;");

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

class ClickableLabel final : public QLabel {
public:
    explicit ClickableLabel(QWidget* parent = nullptr) : QLabel(parent) {}

    std::function<void(const QPoint&)> on_click;
    std::function<void(const QPoint&)> on_press;
    std::function<void(const QPoint&)> on_move;
    std::function<void(const QPoint&)> on_release;

protected:
    void mousePressEvent(QMouseEvent* event) override {
        if (on_press) {
            on_press(event->pos());
        }
        if (on_click) {
            on_click(event->pos());
        }
        QLabel::mousePressEvent(event);
    }

    void mouseMoveEvent(QMouseEvent* event) override {
        if (on_move) {
            on_move(event->pos());
        }
        QLabel::mouseMoveEvent(event);
    }

    void mouseReleaseEvent(QMouseEvent* event) override {
        if (on_release) {
            on_release(event->pos());
        }
        QLabel::mouseReleaseEvent(event);
    }
};

static std::vector<int> enumerate_video_indices_linux() {
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

class IntrinsicsTab final : public QWidget {
public:
    explicit IntrinsicsTab(QWidget* parent = nullptr) : QWidget(parent) {
        auto* root_layout = new QHBoxLayout(this);
        root_layout->setSpacing(14);
        root_layout->setContentsMargins(10, 10, 10, 10);
        setObjectName("intrinsicsTabRoot");

        auto* left_panel = new QGroupBox("Intrinsics Inputs");
        auto* left_form = new QFormLayout(left_panel);
        left_form->setVerticalSpacing(12);
        left_form->setHorizontalSpacing(12);
        left_form->setLabelAlignment(Qt::AlignLeft);
        left_form->setFieldGrowthPolicy(QFormLayout::ExpandingFieldsGrow);

        camera_combo_ = new QComboBox();
        refresh_cameras_btn_ = new QPushButton("Refresh");
        auto* cam_row = new QWidget();
        auto* cam_row_layout = new QHBoxLayout(cam_row);
        cam_row_layout->setContentsMargins(0, 0, 0, 0);
        cam_row_layout->addWidget(camera_combo_, 1);
        cam_row_layout->addWidget(refresh_cameras_btn_);

        resolution_combo_ = new QComboBox();
        refresh_resolutions_btn_ = new QPushButton("Scan Modes");
        auto* resolution_row = new QWidget();
        auto* resolution_row_layout = new QHBoxLayout(resolution_row);
        resolution_row_layout->setContentsMargins(0, 0, 0, 0);
        resolution_row_layout->addWidget(resolution_combo_, 1);
        resolution_row_layout->addWidget(refresh_resolutions_btn_);

        corners_x_ = new QSpinBox();
        corners_x_->setRange(3, 30);
        corners_x_->setValue(10);

        corners_y_ = new QSpinBox();
        corners_y_->setRange(3, 30);
        corners_y_->setValue(7);

        square_size_ = new QLineEdit("0.023");

        save_path_ = new QLineEdit("intrinsics.yaml");
        save_path_->setMinimumWidth(140);
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
        left_form->addRow("Resolution", resolution_row);
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
        calibrate_button_->setObjectName("primaryButton");
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
        set_state_label(auto_state_label_, "Auto-capture OFF", "#f14c4c");

        auto* coverage_panel = new QWidget();
        auto* coverage_form = new QFormLayout(coverage_panel);
        coverage_form->setContentsMargins(0, 0, 12, 0);
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

        auto* left_scroll = new QScrollArea();
        left_scroll->setObjectName("sidebarScroll");
        left_scroll->setWidget(left_container);
        left_scroll->setWidgetResizable(true);
        left_scroll->setFrameShape(QFrame::NoFrame);
        left_scroll->setHorizontalScrollBarPolicy(Qt::ScrollBarAlwaysOff);
        left_scroll->setMinimumWidth(440);

        auto* right_panel = new QGroupBox("View Finder");
        auto* right_layout = new QVBoxLayout(right_panel);
        preview_label_ = new QLabel("Preview stopped.");
        preview_label_->setAlignment(Qt::AlignCenter);
        preview_label_->setMinimumSize(640, 400);
        preview_label_->setObjectName("previewArea");
        right_layout->addWidget(preview_label_);

        root_layout->addWidget(left_scroll, 1);
        root_layout->addWidget(right_panel, 3);

        timer_ = new QTimer(this);
        timer_->setInterval(33);

        connect(refresh_cameras_btn_, &QPushButton::clicked, this, [this]() { refresh_cameras(); });
        connect(refresh_resolutions_btn_, &QPushButton::clicked, this, [this]() { refresh_resolutions(); });
        connect(camera_combo_, qOverload<int>(&QComboBox::currentIndexChanged), this, [this](int) {
            refresh_resolutions();
        });
        connect(resolution_combo_, qOverload<int>(&QComboBox::currentIndexChanged), this, [this](int) {
            apply_selected_resolution();
        });
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
        set_state_label(auto_state_label_, "Auto-capture ON", "#89d185");
        info_banner_->setText("Auto-capture is active. Captured frames will be saved with the calibration output.");
        status_label_->setText("Calibration mode started. Valid detections are auto-captured.");
        set_toggle_button_visual(auto_capture_btn_, true);
        apply_ui_state();
    }

    void stop_auto_capture() {
        auto_capture_enabled_ = false;
        set_state_label(auto_state_label_, "Auto-capture OFF", "#f14c4c");
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

    static std::vector<cv::Size> candidate_resolutions() {
        return {
            cv::Size(320, 240),
            cv::Size(640, 480),
            cv::Size(800, 600),
            cv::Size(960, 540),
            cv::Size(1024, 576),
            cv::Size(1024, 768),
            cv::Size(1280, 720),
            cv::Size(1280, 800),
            cv::Size(1280, 960),
            cv::Size(1600, 900),
            cv::Size(1920, 1080)
        };
    }

    std::vector<cv::Size> probe_camera_resolutions(int device_idx) const {
        std::vector<cv::Size> supported;
        cv::VideoCapture probe(device_idx, cv::CAP_V4L2);
        if (!probe.isOpened()) {
            probe.open(device_idx, cv::CAP_ANY);
        }
        if (!probe.isOpened()) {
            return supported;
        }

        probe.set(cv::CAP_PROP_FOURCC, cv::VideoWriter::fourcc('M', 'J', 'P', 'G'));
        std::set<std::pair<int, int>> unique_modes;
        for (const auto& option : candidate_resolutions()) {
            probe.set(cv::CAP_PROP_FRAME_WIDTH, option.width);
            probe.set(cv::CAP_PROP_FRAME_HEIGHT, option.height);

            const int actual_w = static_cast<int>(std::lround(probe.get(cv::CAP_PROP_FRAME_WIDTH)));
            const int actual_h = static_cast<int>(std::lround(probe.get(cv::CAP_PROP_FRAME_HEIGHT)));
            if (actual_w <= 0 || actual_h <= 0) {
                continue;
            }

            const bool close_to_requested =
                std::abs(actual_w - option.width) <= 16 && std::abs(actual_h - option.height) <= 16;
            if (!close_to_requested) {
                continue;
            }

            if (unique_modes.insert({actual_w, actual_h}).second) {
                supported.emplace_back(actual_w, actual_h);
            }
        }

        std::sort(supported.begin(), supported.end(), [](const cv::Size& a, const cv::Size& b) {
            if (a.width * a.height == b.width * b.height) {
                return a.width < b.width;
            }
            return a.width * a.height < b.width * b.height;
        });
        return supported;
    }

    void refresh_resolutions() {
        resolution_combo_->blockSignals(true);
        resolution_combo_->clear();
        resolution_combo_->addItem("Default (driver)", -1);

        if (camera_combo_->count() == 0) {
            resolution_combo_->setEnabled(false);
            refresh_resolutions_btn_->setEnabled(false);
            resolution_combo_->blockSignals(false);
            return;
        }

        const int device_idx = camera_combo_->currentData().toInt();
        const auto supported = probe_camera_resolutions(device_idx);
        for (const auto& mode : supported) {
            const QString label = QString("%1 x %2").arg(mode.width).arg(mode.height);
            resolution_combo_->addItem(label, (mode.width << 16) | mode.height);
        }

        int preferred_idx = 0;
        for (int i = 1; i < resolution_combo_->count(); ++i) {
            if (resolution_combo_->itemText(i).startsWith("1280 x 720")) {
                preferred_idx = i;
                break;
            }
        }
        resolution_combo_->setCurrentIndex(preferred_idx);
        resolution_combo_->setEnabled(true);
        refresh_resolutions_btn_->setEnabled(true);
        resolution_combo_->blockSignals(false);
    }

    void apply_selected_resolution() {
        if (!cap_.isOpened() || resolution_combo_->count() == 0) {
            return;
        }

        const int packed = resolution_combo_->currentData().toInt();
        if (packed > 0) {
            const int width = (packed >> 16) & 0xFFFF;
            const int height = packed & 0xFFFF;
            // Raw YUYV is USB-bandwidth-limited on most UVC cams (often capped at 640x480);
            // MJPG must be selected before width/height to unlock 720p/1080p modes, matching
            // the negotiation used by probe_camera_resolutions().
            cap_.set(cv::CAP_PROP_FOURCC, cv::VideoWriter::fourcc('M', 'J', 'P', 'G'));
            cap_.set(cv::CAP_PROP_FRAME_WIDTH, width);
            cap_.set(cv::CAP_PROP_FRAME_HEIGHT, height);
        }

        const int actual_w = static_cast<int>(std::lround(cap_.get(cv::CAP_PROP_FRAME_WIDTH)));
        const int actual_h = static_cast<int>(std::lround(cap_.get(cv::CAP_PROP_FRAME_HEIGHT)));
        if (actual_w > 0 && actual_h > 0) {
            status_label_->setText(QString("Active capture resolution: %1 x %2").arg(actual_w).arg(actual_h));
        }
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
            refresh_resolutions();
        } else {
            status_label_->setText(QString("Found %1 camera(s).").arg(camera_combo_->count()));
            refresh_resolutions();
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

        apply_selected_resolution();

        preview_btn_->blockSignals(true);
        preview_btn_->setChecked(true);
        preview_btn_->blockSignals(false);
        set_toggle_button_visual(preview_btn_, true);

        timer_->start();
        const int actual_w = static_cast<int>(std::lround(cap_.get(cv::CAP_PROP_FRAME_WIDTH)));
        const int actual_h = static_cast<int>(std::lround(cap_.get(cv::CAP_PROP_FRAME_HEIGHT)));
        status_label_->setText(
            QString("Preview started on camera %1 at %2 x %3.").arg(device_idx).arg(actual_w).arg(actual_h));
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
        resolution_combo_->setEnabled(!preview_on && camera_combo_->count() > 0);
        refresh_resolutions_btn_->setEnabled(!preview_on && camera_combo_->count() > 0);
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
    QComboBox* resolution_combo_ = nullptr;
    QPushButton* refresh_cameras_btn_ = nullptr;
    QPushButton* refresh_resolutions_btn_ = nullptr;
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

class HomographyTab final : public QWidget {
public:
    explicit HomographyTab(QWidget* parent = nullptr) : QWidget(parent) {
        auto* root_layout = new QHBoxLayout(this);
        root_layout->setSpacing(14);
        root_layout->setContentsMargins(10, 10, 10, 10);

        auto* left_panel = new QGroupBox("Homography Calibration");
        auto* left_form = new QFormLayout(left_panel);
        left_form->setVerticalSpacing(12);
        left_form->setHorizontalSpacing(12);
        left_form->setLabelAlignment(Qt::AlignLeft);
        left_form->setFieldGrowthPolicy(QFormLayout::ExpandingFieldsGrow);

        camera_combo_ = new QComboBox();
        refresh_btn_ = new QPushButton("Refresh");
        auto* cam_row = new QWidget();
        auto* cam_layout = new QHBoxLayout(cam_row);
        cam_layout->setContentsMargins(0, 0, 0, 0);
        cam_layout->addWidget(camera_combo_, 1);
        cam_layout->addWidget(refresh_btn_);

        intrinsics_path_ = new QLineEdit("intrinsics.yaml");
        intrinsics_path_->setMinimumWidth(110);
        auto* browse_intrinsics_btn = new QPushButton("Browse");
        auto* reload_intrinsics_btn = new QPushButton("Reload");
        auto* intrinsics_row = new QWidget();
        auto* intrinsics_layout = new QHBoxLayout(intrinsics_row);
        intrinsics_layout->setContentsMargins(0, 0, 0, 0);
        intrinsics_layout->addWidget(intrinsics_path_, 1);
        intrinsics_layout->addWidget(browse_intrinsics_btn);
        intrinsics_layout->addWidget(reload_intrinsics_btn);

        mode_combo_ = new QComboBox();
        mode_combo_->addItem("Manual 4 Points");
        mode_combo_->addItem("Auto ArUco 4 Markers");

        width_m_ = new QLineEdit("2.0");
        height_m_ = new QLineEdit("2.0");
        ground_m_ = new QLineEdit("1.5");

        auto* cam_dist_row = new QWidget();
        auto* cam_dist_layout = new QGridLayout(cam_dist_row);
        cam_dist_layout->setContentsMargins(0, 0, 0, 0);
        cam_dist_layout->setHorizontalSpacing(6);
        cam_dist_layout->setVerticalSpacing(6);
        cam_tl_m_ = new QLineEdit("2.8");
        cam_tr_m_ = new QLineEdit("2.8");
        cam_br_m_ = new QLineEdit("4.0");
        cam_bl_m_ = new QLineEdit("4.0");
        for (auto* edit : {cam_tl_m_, cam_tr_m_, cam_br_m_, cam_bl_m_}) {
            edit->setMaximumWidth(80);
        }
        cam_dist_layout->addWidget(new QLabel("TL"), 0, 0);
        cam_dist_layout->addWidget(cam_tl_m_, 0, 1);
        cam_dist_layout->addWidget(new QLabel("TR"), 0, 2);
        cam_dist_layout->addWidget(cam_tr_m_, 0, 3);
        cam_dist_layout->addWidget(new QLabel("BR"), 1, 0);
        cam_dist_layout->addWidget(cam_br_m_, 1, 1);
        cam_dist_layout->addWidget(new QLabel("BL"), 1, 2);
        cam_dist_layout->addWidget(cam_bl_m_, 1, 3);

        save_path_ = new QLineEdit("homography.yaml");
        save_path_->setMinimumWidth(140);
        auto* browse_btn = new QPushButton("Browse");
        auto* save_row = new QWidget();
        auto* save_layout = new QHBoxLayout(save_row);
        save_layout->setContentsMargins(0, 0, 0, 0);
        save_layout->addWidget(save_path_, 1);
        save_layout->addWidget(browse_btn);

        auto* ids_row = new QWidget();
        auto* ids_layout = new QGridLayout(ids_row);
        ids_layout->setContentsMargins(0, 0, 0, 0);
        ids_layout->setHorizontalSpacing(6);
        ids_layout->setVerticalSpacing(6);
        id_tl_ = new QSpinBox();
        id_tr_ = new QSpinBox();
        id_br_ = new QSpinBox();
        id_bl_ = new QSpinBox();
        for (auto* box : {id_tl_, id_tr_, id_br_, id_bl_}) {
            box->setRange(0, 1024);
            box->setMaximumWidth(80);
        }
        id_tl_->setValue(0);
        id_tr_->setValue(1);
        id_br_->setValue(2);
        id_bl_->setValue(3);
        ids_layout->addWidget(new QLabel("TL"), 0, 0);
        ids_layout->addWidget(id_tl_, 0, 1);
        ids_layout->addWidget(new QLabel("TR"), 0, 2);
        ids_layout->addWidget(id_tr_, 0, 3);
        ids_layout->addWidget(new QLabel("BR"), 1, 0);
        ids_layout->addWidget(id_br_, 1, 1);
        ids_layout->addWidget(new QLabel("BL"), 1, 2);
        ids_layout->addWidget(id_bl_, 1, 3);

        left_form->addRow("Camera", cam_row);
        left_form->addRow("Intrinsics YAML", intrinsics_row);
        left_form->addRow("Mode", mode_combo_);
        left_form->addRow("Plane Width (m)", width_m_);
        left_form->addRow("Plane Height (m)", height_m_);
        left_form->addRow("Camera to Ground (m)", ground_m_);
        left_form->addRow("Cam->Corner Dist (m, slant)", cam_dist_row);
        left_form->addRow("Aruco IDs (TL/TR/BR/BL)", ids_row);
        left_form->addRow("Save YAML", save_row);

        preview_toggle_ = make_toggle_button("Preview");
        preview_toggle_->setMinimumHeight(34);
        validation_toggle_ = make_toggle_button("Validation");
        validation_toggle_->setMinimumHeight(34);
        detect_btn_ = new QPushButton("Detect 4 Points");
        clear_btn_ = new QPushButton("Clear Points");
        solve_btn_ = new QPushButton("Solve + Save");
        solve_btn_->setObjectName("primaryButton");

        auto* controls = new QGroupBox("Actions");
        auto* controls_layout = new QVBoxLayout(controls);
        controls_layout->setSpacing(10);

        auto* toggle_row = new QWidget();
        auto* toggle_row_layout = new QHBoxLayout(toggle_row);
        toggle_row_layout->setContentsMargins(0, 0, 0, 0);
        toggle_row_layout->addWidget(preview_toggle_);
        toggle_row_layout->addWidget(validation_toggle_);

        auto* action_row = new QWidget();
        auto* action_row_layout = new QHBoxLayout(action_row);
        action_row_layout->setContentsMargins(0, 0, 0, 0);
        action_row_layout->addWidget(detect_btn_);
        action_row_layout->addWidget(clear_btn_);
        action_row_layout->addWidget(solve_btn_);

        controls_layout->addWidget(toggle_row);
        controls_layout->addWidget(action_row);

        points_label_ = new QLabel("Selected points: 0/4");
        distance_label_ = new QLabel("Distance: n/a");
        intrinsics_status_label_ = new QLabel("Intrinsics: not loaded. Load an intrinsics YAML before detecting points.");
        intrinsics_status_label_->setWordWrap(true);
        status_label_ = new QLabel("Manual mode: click 4 points in order TL, TR, BR, BL.");
        status_label_->setWordWrap(true);

        auto* left_stack = new QVBoxLayout();
        left_stack->addWidget(left_panel);
        left_stack->addWidget(intrinsics_status_label_);
        left_stack->addWidget(controls);
        left_stack->addWidget(points_label_);
        left_stack->addWidget(distance_label_);
        left_stack->addWidget(status_label_);
        left_stack->addStretch(1);

        auto* left_container = new QWidget();
        left_container->setLayout(left_stack);

        auto* left_scroll = new QScrollArea();
        left_scroll->setObjectName("sidebarScroll");
        left_scroll->setWidget(left_container);
        left_scroll->setWidgetResizable(true);
        left_scroll->setFrameShape(QFrame::NoFrame);
        left_scroll->setHorizontalScrollBarPolicy(Qt::ScrollBarAlwaysOff);
        left_scroll->setMinimumWidth(440);

        auto* right_panel = new QGroupBox("View Finder");
        auto* right_layout = new QVBoxLayout(right_panel);
        preview_label_ = new ClickableLabel();
        preview_label_->setObjectName("previewArea");
        preview_label_->setAlignment(Qt::AlignCenter);
        preview_label_->setMinimumSize(640, 400);
        preview_label_->setText("Preview stopped.");
        right_layout->addWidget(preview_label_);

        root_layout->addWidget(left_scroll, 1);
        root_layout->addWidget(right_panel, 3);

        timer_ = new QTimer(this);
        timer_->setInterval(33);

        preview_label_->on_click = [this](const QPoint& pos) { on_preview_click(pos); };
        preview_label_->on_press = [this](const QPoint& pos) { on_preview_press(pos); };
        preview_label_->on_move = [this](const QPoint& pos) { on_preview_move(pos); };
        preview_label_->on_release = [this](const QPoint& pos) { on_preview_release(pos); };

        connect(refresh_btn_, &QPushButton::clicked, this, [this]() { refresh_cameras(); });
        connect(browse_intrinsics_btn, &QPushButton::clicked, this, [this]() {
            const QString path = QFileDialog::getOpenFileName(
                this, "Load Intrinsics YAML", intrinsics_path_->text(), "YAML files (*.yaml *.yml)");
            if (!path.isEmpty()) {
                intrinsics_path_->setText(path);
                load_intrinsics();
            }
        });
        connect(reload_intrinsics_btn, &QPushButton::clicked, this, [this]() { load_intrinsics(); });
        connect(preview_toggle_, &QToolButton::toggled, this, [this](bool checked) {
            if (checked) {
                start_preview();
            } else {
                stop_preview();
            }
        });
        connect(validation_toggle_, &QToolButton::toggled, this, [this](bool checked) {
            if (checked) {
                start_validation();
            } else {
                stop_validation();
            }
        });
        connect(detect_btn_, &QPushButton::clicked, this, [this]() { detect_points(); });
        connect(clear_btn_, &QPushButton::clicked, this, [this]() { clear_points(); });
        connect(solve_btn_, &QPushButton::clicked, this, [this]() { solve_and_save(); });
        connect(mode_combo_, &QComboBox::currentTextChanged, this, [this]() {
            clear_points();
            status_label_->setText(mode_combo_->currentIndex() == 0
                ? "Manual mode: click 4 points in order TL, TR, BR, BL."
                : "Auto mode: click Detect 4 Points to read ArUco IDs and corners.");
        });
        connect(browse_btn, &QPushButton::clicked, this, [this]() {
            const QString path = QFileDialog::getSaveFileName(
                this,
                "Save Homography YAML",
                save_path_->text(),
                "YAML files (*.yaml *.yml)");
            if (!path.isEmpty()) {
                save_path_->setText(path);
            }
        });
        connect(timer_, &QTimer::timeout, this, [this]() { on_frame_tick(); });

        refresh_cameras();
        apply_ui_state();
    }

    ~HomographyTab() override {
        stop_preview();
    }

private:
    std::vector<cv::Point2f> world_rect_points() const {
        bool ok_w = false;
        bool ok_h = false;
        const double w = width_m_->text().toDouble(&ok_w);
        const double h = height_m_->text().toDouble(&ok_h);
        if (!ok_w || !ok_h || w <= 0.0 || h <= 0.0) {
            return {};
        }
        return {
            cv::Point2f(0.f, 0.f),
            cv::Point2f(static_cast<float>(w), 0.f),
            cv::Point2f(static_cast<float>(w), static_cast<float>(h)),
            cv::Point2f(0.f, static_cast<float>(h))
        };
    }

    bool parse_camera_corner_slant_m(std::array<double, 4>& slant_m, QString& error) const {
        bool ok_tl = false;
        bool ok_tr = false;
        bool ok_br = false;
        bool ok_bl = false;

        slant_m[0] = cam_tl_m_->text().toDouble(&ok_tl);
        slant_m[1] = cam_tr_m_->text().toDouble(&ok_tr);
        slant_m[2] = cam_br_m_->text().toDouble(&ok_br);
        slant_m[3] = cam_bl_m_->text().toDouble(&ok_bl);

        if (!(ok_tl && ok_tr && ok_br && ok_bl)) {
            error = "Camera-to-corner distances must be numeric.";
            return false;
        }
        for (double v : slant_m) {
            if (v <= 0.0) {
                error = "Camera-to-corner distances must be positive.";
                return false;
            }
        }
        return true;
    }

    bool estimate_camera_ground_xy(cv::Point2f& camera_xy, std::array<double, 4>& planar_m, QString& error) const {
        const auto world = world_rect_points();
        if (world.size() != 4) {
            error = "Invalid world plane dimensions.";
            return false;
        }

        std::array<double, 4> slant_m{};
        if (!parse_camera_corner_slant_m(slant_m, error)) {
            return false;
        }

        bool ok_ground = false;
        const double ground_h = ground_m_->text().toDouble(&ok_ground);
        if (!ok_ground || ground_h < 0.0) {
            error = "Camera to ground must be >= 0.";
            return false;
        }

        for (size_t i = 0; i < slant_m.size(); ++i) {
            const double d = slant_m[i];
            if (d <= ground_h) {
                error = "Each cam->corner slant distance must be greater than camera-to-ground height.";
                return false;
            }
            planar_m[i] = std::sqrt(std::max(0.0, d * d - ground_h * ground_h));
        }

        const cv::Point2f p0 = world[0];
        const double r0 = planar_m[0];
        cv::Mat A(3, 2, CV_64F);
        cv::Mat b(3, 1, CV_64F);

        for (int i = 1; i < 4; ++i) {
            const cv::Point2f pi = world[i];
            const double ri = planar_m[i];
            A.at<double>(i - 1, 0) = 2.0 * (pi.x - p0.x);
            A.at<double>(i - 1, 1) = 2.0 * (pi.y - p0.y);
            b.at<double>(i - 1, 0) = (r0 * r0 - ri * ri) - (p0.x * p0.x - pi.x * pi.x) - (p0.y * p0.y - pi.y * pi.y);
        }

        cv::Mat x;
        if (!cv::solve(A, b, x, cv::DECOMP_SVD)) {
            error = "Failed to solve camera ground position from distance constraints.";
            return false;
        }

        camera_xy = cv::Point2f(static_cast<float>(x.at<double>(0, 0)), static_cast<float>(x.at<double>(1, 0)));
        return std::isfinite(camera_xy.x) && std::isfinite(camera_xy.y);
    }

    void load_intrinsics() {
        IntrinsicsData data;
        QString error;
        if (!load_intrinsics_yaml(intrinsics_path_->text().toStdString(), data, error)) {
            intrinsics_loaded_ = false;
            intrinsics_status_label_->setText(QString("Intrinsics: failed to load (%1)").arg(error));
            apply_ui_state();
            return;
        }

        intrinsics_image_size_ = cv::Size(data.image_width, data.image_height);
        intrinsics_camera_matrix_ = data.camera_matrix;
        intrinsics_dist_coeffs_ = data.dist_coeffs;
        intrinsics_projection_matrix_ = data.projection_matrix;
        intrinsics_loaded_ = true;

        intrinsics_status_label_->setText(QString("Intrinsics loaded: %1x%2, fx=%3, fy=%4")
            .arg(data.image_width)
            .arg(data.image_height)
            .arg(intrinsics_camera_matrix_.at<double>(0, 0), 0, 'f', 2)
            .arg(intrinsics_camera_matrix_.at<double>(1, 1), 0, 'f', 2));

        if (cap_.isOpened()) {
            status_label_->setText("Intrinsics loaded. Restart preview to apply the new resolution/rectification.");
        } else {
            status_label_->setText("Intrinsics loaded. Start preview to capture at this resolution with rectification applied.");
        }
        apply_ui_state();
    }

    void refresh_cameras() {
        camera_combo_->clear();
        const auto candidates = enumerate_video_indices_linux();
        for (const int idx : candidates) {
            cv::VideoCapture probe(idx, cv::CAP_V4L2);
            if (probe.isOpened()) {
                camera_combo_->addItem(QString("Camera %1").arg(idx), idx);
                probe.release();
            }
        }
        if (camera_combo_->count() == 0) {
            status_label_->setText("No cameras found.");
        }
    }

    void start_preview() {
        if (camera_combo_->count() == 0) {
            status_label_->setText("No camera selected.");
            preview_toggle_->blockSignals(true);
            preview_toggle_->setChecked(false);
            preview_toggle_->blockSignals(false);
            return;
        }

        const int idx = camera_combo_->currentData().toInt();
        cap_.open(idx, cv::CAP_V4L2);
        if (!cap_.isOpened()) {
            cap_.open(idx, cv::CAP_ANY);
        }
        if (!cap_.isOpened()) {
            status_label_->setText(QString("Failed to open camera %1.").arg(idx));
            preview_toggle_->blockSignals(true);
            preview_toggle_->setChecked(false);
            preview_toggle_->blockSignals(false);
            return;
        }

        if (intrinsics_loaded_) {
            // MJPG is required to unlock the intrinsics' capture resolution on most UVC cameras --
            // see the same fix applied to IntrinsicsTab::apply_selected_resolution().
            cap_.set(cv::CAP_PROP_FOURCC, cv::VideoWriter::fourcc('M', 'J', 'P', 'G'));
            cap_.set(cv::CAP_PROP_FRAME_WIDTH, intrinsics_image_size_.width);
            cap_.set(cv::CAP_PROP_FRAME_HEIGHT, intrinsics_image_size_.height);
        }

        timer_->start();
        set_toggle_button_visual(preview_toggle_, true);

        if (intrinsics_loaded_) {
            const int actual_w = static_cast<int>(std::lround(cap_.get(cv::CAP_PROP_FRAME_WIDTH)));
            const int actual_h = static_cast<int>(std::lround(cap_.get(cv::CAP_PROP_FRAME_HEIGHT)));
            if (actual_w != intrinsics_image_size_.width || actual_h != intrinsics_image_size_.height) {
                status_label_->setText(QString(
                    "Warning: camera gave %1x%2 but intrinsics were calibrated at %3x%4. "
                    "Rectification/homography will be inaccurate -- recalibrate intrinsics at this resolution.")
                    .arg(actual_w).arg(actual_h)
                    .arg(intrinsics_image_size_.width).arg(intrinsics_image_size_.height));
            } else {
                status_label_->setText(QString("Preview started at %1x%2 (matches intrinsics).").arg(actual_w).arg(actual_h));
            }
        }
        apply_ui_state();
    }

    void stop_preview() {
        stop_validation();
        if (timer_->isActive()) {
            timer_->stop();
        }
        if (cap_.isOpened()) {
            cap_.release();
        }
        preview_toggle_->blockSignals(true);
        preview_toggle_->setChecked(false);
        preview_toggle_->blockSignals(false);
        set_toggle_button_visual(preview_toggle_, false);
        preview_label_->setText("Preview stopped.");
        apply_ui_state();
    }

    void start_validation() {
        if (!cap_.isOpened()) {
            status_label_->setText("Start preview before enabling validation.");
            validation_toggle_->blockSignals(true);
            validation_toggle_->setChecked(false);
            validation_toggle_->blockSignals(false);
            set_toggle_button_visual(validation_toggle_, false);
            return;
        }
        if (homography_.empty()) {
            status_label_->setText("Solve homography first, then enable validation mode.");
            validation_toggle_->blockSignals(true);
            validation_toggle_->setChecked(false);
            validation_toggle_->blockSignals(false);
            set_toggle_button_visual(validation_toggle_, false);
            return;
        }
        if (!camera_origin_valid_) {
            status_label_->setText("Camera origin is not solved. Check cam->corner distances and solve again.");
            validation_toggle_->blockSignals(true);
            validation_toggle_->setChecked(false);
            validation_toggle_->blockSignals(false);
            set_toggle_button_visual(validation_toggle_, false);
            return;
        }

        validation_enabled_ = true;
        status_label_->setText("Validation ON: drag a bbox around an object. Distance uses bbox bottom-center on ground plane.");
        set_toggle_button_visual(validation_toggle_, true);
        apply_ui_state();
    }

    void stop_validation() {
        validation_enabled_ = false;
        dragging_bbox_ = false;
        bbox_valid_ = false;
        distance_label_->setText("Distance: n/a");
        set_toggle_button_visual(validation_toggle_, false);
        apply_ui_state();
    }

    QPointF label_to_frame(const QPoint& p) const {
        if (current_frame_.empty()) {
            return QPointF(-1, -1);
        }

        const QSize widget_size = preview_label_->size();
        const double sx = static_cast<double>(widget_size.width()) / current_frame_.cols;
        const double sy = static_cast<double>(widget_size.height()) / current_frame_.rows;
        const double scale = std::min(sx, sy);
        const int draw_w = static_cast<int>(current_frame_.cols * scale);
        const int draw_h = static_cast<int>(current_frame_.rows * scale);
        const int off_x = (widget_size.width() - draw_w) / 2;
        const int off_y = (widget_size.height() - draw_h) / 2;

        if (p.x() < off_x || p.y() < off_y || p.x() >= off_x + draw_w || p.y() >= off_y + draw_h) {
            return QPointF(-1, -1);
        }

        const double x = (p.x() - off_x) / scale;
        const double y = (p.y() - off_y) / scale;
        return QPointF(x, y);
    }

    bool compute_distance_from_bbox() {
        if (homography_.empty() || !bbox_valid_ || !camera_origin_valid_) {
            return false;
        }
        const float x0 = std::min(bbox_p0_.x, bbox_p1_.x);
        const float y0 = std::min(bbox_p0_.y, bbox_p1_.y);
        const float x1 = std::max(bbox_p0_.x, bbox_p1_.x);
        const float y1 = std::max(bbox_p0_.y, bbox_p1_.y);

        const cv::Point2f foot((x0 + x1) * 0.5f, y1);
        std::vector<cv::Point2f> image_pts = {foot};
        std::vector<cv::Point2f> world_pts;
        cv::perspectiveTransform(image_pts, world_pts, homography_);
        if (world_pts.empty()) {
            return false;
        }

        last_world_pt_ = world_pts[0];
        const double dx = static_cast<double>(last_world_pt_.x - camera_ground_xy_.x);
        const double dy = static_cast<double>(last_world_pt_.y - camera_ground_xy_.y);
        last_distance_m_ = std::sqrt(dx * dx + dy * dy);
        distance_label_->setText(QString("Distance: %1 m (x=%2, y=%3)")
            .arg(last_distance_m_, 0, 'f', 2)
            .arg(last_world_pt_.x, 0, 'f', 2)
            .arg(last_world_pt_.y, 0, 'f', 2));
        return true;
    }

    void on_preview_press(const QPoint& pos) {
        if (!validation_enabled_ || current_frame_.empty()) {
            return;
        }
        const QPointF p = label_to_frame(pos);
        if (p.x() < 0 || p.y() < 0) {
            return;
        }
        dragging_bbox_ = true;
        bbox_valid_ = false;
        bbox_p0_ = cv::Point2f(static_cast<float>(p.x()), static_cast<float>(p.y()));
        bbox_p1_ = bbox_p0_;
    }

    void on_preview_move(const QPoint& pos) {
        if (!validation_enabled_ || !dragging_bbox_) {
            return;
        }
        const QPointF p = label_to_frame(pos);
        if (p.x() < 0 || p.y() < 0) {
            return;
        }
        bbox_p1_ = cv::Point2f(static_cast<float>(p.x()), static_cast<float>(p.y()));
    }

    void on_preview_release(const QPoint& pos) {
        if (!validation_enabled_ || !dragging_bbox_) {
            return;
        }
        dragging_bbox_ = false;
        const QPointF p = label_to_frame(pos);
        if (p.x() < 0 || p.y() < 0) {
            return;
        }
        bbox_p1_ = cv::Point2f(static_cast<float>(p.x()), static_cast<float>(p.y()));

        const float w = std::abs(bbox_p1_.x - bbox_p0_.x);
        const float h = std::abs(bbox_p1_.y - bbox_p0_.y);
        bbox_valid_ = (w >= 8.0f && h >= 8.0f);
        if (!bbox_valid_) {
            status_label_->setText("Validation bbox too small. Draw a larger box.");
            return;
        }

        if (!compute_distance_from_bbox()) {
            status_label_->setText("Failed to estimate distance from bbox.");
            return;
        }
        status_label_->setText(QString("Estimated distance: %1 m. Compare this with tape measurement.")
            .arg(last_distance_m_, 0, 'f', 2));
    }

    void on_preview_click(const QPoint& pos) {
        if (validation_enabled_ || mode_combo_->currentIndex() != 0 || !cap_.isOpened()) {
            return;
        }
        if (!intrinsics_loaded_) {
            status_label_->setText("Load an intrinsics YAML before selecting points -- homography must be solved on the rectified image.");
            return;
        }

        const QPointF mapped = label_to_frame(pos);
        if (mapped.x() < 0 || mapped.y() < 0) {
            return;
        }
        if (manual_points_.size() >= 4) {
            return;
        }

        manual_points_.push_back(cv::Point2f(static_cast<float>(mapped.x()), static_cast<float>(mapped.y())));
        image_points_ = manual_points_;
        points_label_->setText(QString("Selected points: %1/4").arg(manual_points_.size()));
        status_label_->setText("Manual point added.");
        apply_ui_state();
    }

    void detect_points() {
        if (!cap_.isOpened() || current_frame_.empty()) {
            status_label_->setText("Start preview before detection.");
            return;
        }
        if (!intrinsics_loaded_) {
            status_label_->setText("Load an intrinsics YAML before detecting points -- homography must be solved on the rectified image.");
            return;
        }

        if (mode_combo_->currentIndex() == 0) {
            status_label_->setText("Manual mode: click 4 points on the image.");
            return;
        }

        cv::aruco::Dictionary dict_raw = cv::aruco::getPredefinedDictionary(cv::aruco::DICT_4X4_50);
        cv::Ptr<cv::aruco::Dictionary> dict = cv::makePtr<cv::aruco::Dictionary>(dict_raw);
        std::vector<std::vector<cv::Point2f>> corners;
        std::vector<int> ids;
        cv::aruco::detectMarkers(current_frame_, dict, corners, ids);

        if (ids.empty()) {
            status_label_->setText("No ArUco markers detected.");
            return;
        }

        const std::array<int, 4> wanted = {
            id_tl_->value(), id_tr_->value(), id_br_->value(), id_bl_->value()
        };
        std::array<cv::Point2f, 4> centers{};
        std::array<bool, 4> found = {false, false, false, false};

        for (size_t i = 0; i < ids.size(); ++i) {
            for (int k = 0; k < 4; ++k) {
                if (ids[i] == wanted[k]) {
                    cv::Point2f c(0.f, 0.f);
                    for (const auto& p : corners[i]) {
                        c += p;
                    }
                    c *= 0.25f;
                    centers[k] = c;
                    found[k] = true;
                }
            }
        }

        if (!std::all_of(found.begin(), found.end(), [](bool v) { return v; })) {
            status_label_->setText("Could not find all four configured marker IDs.");
            return;
        }

        image_points_ = {centers[0], centers[1], centers[2], centers[3]};
        points_label_->setText("Selected points: 4/4");
        status_label_->setText("Auto 4-point detection successful.");
        apply_ui_state();
    }

    void clear_points() {
        stop_validation();
        manual_points_.clear();
        image_points_.clear();
        homography_.release();
        camera_origin_valid_ = false;
        points_label_->setText("Selected points: 0/4");
        distance_label_->setText("Distance: n/a");
        apply_ui_state();
    }

    void solve_and_save() {
        if (image_points_.size() != 4) {
            status_label_->setText("Need exactly 4 image points.");
            return;
        }
        if (!intrinsics_loaded_) {
            status_label_->setText("Load an intrinsics YAML before solving -- homography must use rectified points.");
            return;
        }

        const auto world = world_rect_points();
        if (world.size() != 4) {
            status_label_->setText("Invalid plane width/height.");
            return;
        }

        homography_ = cv::findHomography(image_points_, world, cv::RANSAC);
        if (homography_.empty()) {
            status_label_->setText("Homography solve failed.");
            return;
        }

        std::array<double, 4> camera_corner_ground_m{};
        QString camera_origin_error;
        if (!estimate_camera_ground_xy(camera_ground_xy_, camera_corner_ground_m, camera_origin_error)) {
            camera_origin_valid_ = false;
            status_label_->setText(QString("Homography solved, but camera ground origin failed: %1").arg(camera_origin_error));
            apply_ui_state();
            return;
        }
        camera_origin_valid_ = true;

        const std::filesystem::path out(save_path_->text().toStdString());
        try {
            if (out.has_parent_path() && !out.parent_path().empty()) {
                std::filesystem::create_directories(out.parent_path());
            }
        } catch (const std::exception&) {
            status_label_->setText("Invalid output path.");
            return;
        }

        std::ofstream fs(out.string(), std::ios::out | std::ios::trunc);
        if (!fs.is_open()) {
            status_label_->setText("Failed to open output YAML path.");
            return;
        }

        fs << std::fixed << std::setprecision(6);
        fs << "image_width: " << current_frame_.cols << "\n";
        fs << "image_height: " << current_frame_.rows << "\n";
        fs << "mode: " << (mode_combo_->currentIndex() == 0 ? "manual" : "aruco_auto") << "\n";
        fs << "camera_to_ground_m: " << ground_m_->text().toStdString() << "\n";
        fs << "plane_width_m: " << width_m_->text().toStdString() << "\n";
        fs << "plane_height_m: " << height_m_->text().toStdString() << "\n";
          fs << "camera_ground_xy_m: [" << camera_ground_xy_.x << ", " << camera_ground_xy_.y << "]\n";
          fs << "camera_to_corner_slant_m: ["
              << cam_tl_m_->text().toStdString() << ", "
              << cam_tr_m_->text().toStdString() << ", "
              << cam_br_m_->text().toStdString() << ", "
              << cam_bl_m_->text().toStdString() << "]\n";
          fs << "camera_to_corner_ground_m: ["
              << camera_corner_ground_m[0] << ", "
              << camera_corner_ground_m[1] << ", "
              << camera_corner_ground_m[2] << ", "
              << camera_corner_ground_m[3] << "]\n";

        // Intrinsics used to rectify the image before this homography was solved -- points were
        // picked on the undistorted frame, so any consumer must undistort with these same
        // parameters (at this same resolution) before applying homography_matrix.
        fs << "intrinsics_source: \"" << intrinsics_path_->text().toStdString() << "\"\n";
        fs << "camera_matrix:\n";
        fs << "  rows: 3\n";
        fs << "  cols: 3\n";
        fs << "  data: ["
           << intrinsics_camera_matrix_.at<double>(0, 0) << ", " << intrinsics_camera_matrix_.at<double>(0, 1) << ", " << intrinsics_camera_matrix_.at<double>(0, 2) << ",\n"
           << "         " << intrinsics_camera_matrix_.at<double>(1, 0) << ", " << intrinsics_camera_matrix_.at<double>(1, 1) << ", " << intrinsics_camera_matrix_.at<double>(1, 2) << ",\n"
           << "         " << intrinsics_camera_matrix_.at<double>(2, 0) << ", " << intrinsics_camera_matrix_.at<double>(2, 1) << ", " << intrinsics_camera_matrix_.at<double>(2, 2) << "]\n";
        fs << "distortion_coefficients:\n";
        fs << "  rows: 1\n";
        fs << "  cols: " << intrinsics_dist_coeffs_.total() << "\n";
        fs << "  data: [";
        for (int i = 0; i < static_cast<int>(intrinsics_dist_coeffs_.total()); ++i) {
            fs << (i == 0 ? "" : ", ") << intrinsics_dist_coeffs_.at<double>(i, 0);
        }
        fs << "]\n";
        fs << "homography_matrix:\n";
        fs << "  rows: 3\n";
        fs << "  cols: 3\n";
        fs << "  data: ["
           << homography_.at<double>(0, 0) << ", " << homography_.at<double>(0, 1) << ", " << homography_.at<double>(0, 2) << ",\n"
           << "         " << homography_.at<double>(1, 0) << ", " << homography_.at<double>(1, 1) << ", " << homography_.at<double>(1, 2) << ",\n"
           << "         " << homography_.at<double>(2, 0) << ", " << homography_.at<double>(2, 1) << ", " << homography_.at<double>(2, 2) << "]\n";
        fs.close();

        status_label_->setText(QString("Homography saved: %1 | camera ground XY=(%2, %3)")
            .arg(QString::fromStdString(out.string()))
            .arg(camera_ground_xy_.x, 0, 'f', 2)
            .arg(camera_ground_xy_.y, 0, 'f', 2));
    }

    void on_frame_tick() {
        if (!cap_.isOpened()) {
            return;
        }
        cv::Mat frame;
        if (!cap_.read(frame) || frame.empty()) {
            status_label_->setText("Frame read failed.");
            return;
        }

        if (intrinsics_loaded_) {
            cv::Mat rectified;
            cv::undistort(frame, rectified, intrinsics_camera_matrix_, intrinsics_dist_coeffs_,
                intrinsics_projection_matrix_);
            frame = rectified;
        }

        current_frame_ = frame.clone();
        cv::Mat overlay = frame.clone();

        for (size_t i = 0; i < image_points_.size(); ++i) {
            cv::circle(overlay, image_points_[i], 7, cv::Scalar(30, 220, 30), -1);
            cv::putText(
                overlay,
                std::to_string(i),
                image_points_[i] + cv::Point2f(8.f, -8.f),
                cv::FONT_HERSHEY_SIMPLEX,
                0.6,
                cv::Scalar(20, 230, 230),
                2);
        }
        if (image_points_.size() == 4) {
            for (int i = 0; i < 4; ++i) {
                cv::line(overlay, image_points_[i], image_points_[(i + 1) % 4], cv::Scalar(40, 200, 255), 2);
            }
        }

        if (validation_enabled_ && (dragging_bbox_ || bbox_valid_)) {
            const float x0 = std::min(bbox_p0_.x, bbox_p1_.x);
            const float y0 = std::min(bbox_p0_.y, bbox_p1_.y);
            const float x1 = std::max(bbox_p0_.x, bbox_p1_.x);
            const float y1 = std::max(bbox_p0_.y, bbox_p1_.y);

            cv::rectangle(
                overlay,
                cv::Rect2f(cv::Point2f(x0, y0), cv::Point2f(x1, y1)),
                cv::Scalar(40, 255, 120),
                2);

            const cv::Point2f foot((x0 + x1) * 0.5f, y1);
            cv::circle(overlay, foot, 5, cv::Scalar(0, 220, 255), -1);

            if (bbox_valid_) {
                const std::string text = "dist=" + std::to_string(last_distance_m_).substr(0, 4) + "m";
                cv::putText(
                    overlay,
                    text,
                    cv::Point(static_cast<int>(x0), std::max(20, static_cast<int>(y0) - 8)),
                    cv::FONT_HERSHEY_SIMPLEX,
                    0.7,
                    cv::Scalar(30, 255, 255),
                    2);
            }
        }

        const QImage image = mat_to_qimage(overlay);
        preview_label_->setPixmap(QPixmap::fromImage(image).scaled(
            preview_label_->size(), Qt::KeepAspectRatio, Qt::SmoothTransformation));
    }

    void apply_ui_state() {
        const bool preview_on = cap_.isOpened();
        const bool can_validate = preview_on && !homography_.empty() && camera_origin_valid_;
        set_toggle_button_visual(preview_toggle_, preview_on);
        set_toggle_button_visual(validation_toggle_, validation_enabled_);
        refresh_btn_->setEnabled(!preview_on);
        camera_combo_->setEnabled(!preview_on);
        validation_toggle_->setEnabled(can_validate);
        detect_btn_->setEnabled(preview_on && intrinsics_loaded_ && !validation_enabled_);
        clear_btn_->setEnabled(!validation_enabled_);
        solve_btn_->setEnabled(preview_on && intrinsics_loaded_ && image_points_.size() == 4 && !validation_enabled_);
        mode_combo_->setEnabled(!validation_enabled_);
        id_tl_->setEnabled(!validation_enabled_);
        id_tr_->setEnabled(!validation_enabled_);
        id_br_->setEnabled(!validation_enabled_);
        id_bl_->setEnabled(!validation_enabled_);
    }

private:
    QComboBox* camera_combo_ = nullptr;
    QPushButton* refresh_btn_ = nullptr;
    QComboBox* mode_combo_ = nullptr;
    QLineEdit* width_m_ = nullptr;
    QLineEdit* height_m_ = nullptr;
    QLineEdit* ground_m_ = nullptr;
    QLineEdit* cam_tl_m_ = nullptr;
    QLineEdit* cam_tr_m_ = nullptr;
    QLineEdit* cam_br_m_ = nullptr;
    QLineEdit* cam_bl_m_ = nullptr;
    QLineEdit* save_path_ = nullptr;
    QLineEdit* intrinsics_path_ = nullptr;
    QLabel* intrinsics_status_label_ = nullptr;
    QSpinBox* id_tl_ = nullptr;
    QSpinBox* id_tr_ = nullptr;
    QSpinBox* id_br_ = nullptr;
    QSpinBox* id_bl_ = nullptr;

    QToolButton* preview_toggle_ = nullptr;
    QToolButton* validation_toggle_ = nullptr;
    QPushButton* detect_btn_ = nullptr;
    QPushButton* clear_btn_ = nullptr;
    QPushButton* solve_btn_ = nullptr;
    QLabel* points_label_ = nullptr;
    QLabel* distance_label_ = nullptr;
    QLabel* status_label_ = nullptr;
    ClickableLabel* preview_label_ = nullptr;
    QTimer* timer_ = nullptr;

    cv::VideoCapture cap_;
    cv::Mat current_frame_;
    bool intrinsics_loaded_ = false;
    cv::Size intrinsics_image_size_;
    cv::Mat intrinsics_camera_matrix_;
    cv::Mat intrinsics_dist_coeffs_;
    cv::Mat intrinsics_projection_matrix_;
    std::vector<cv::Point2f> manual_points_;
    std::vector<cv::Point2f> image_points_;
    cv::Mat homography_;
    bool validation_enabled_ = false;
    bool dragging_bbox_ = false;
    bool bbox_valid_ = false;
    cv::Point2f bbox_p0_;
    cv::Point2f bbox_p1_;
    cv::Point2f last_world_pt_;
    cv::Point2f camera_ground_xy_;
    bool camera_origin_valid_ = false;
    double last_distance_m_ = 0.0;
};

// Keep the platform's native UI font (Segoe UI / San Francisco / the desktop's configured sans
// on Linux) rather than naming a family explicitly: on this Qt/fontconfig combination, swapping
// in an explicitly-named family (even one confirmed installed) corrupts QFormLayout's row-height
// computation and makes rows overlap. A size bump on the existing font is safe and sufficient.
static void apply_professional_font(QApplication& app) {
    QFont font = app.font();
    font.setPointSize(10);
    app.setFont(font);
}

int main(int argc, char** argv) {
    QApplication app(argc, argv);
    apply_professional_font(app);

    // VS Code Dark+ palette.
    app.setStyleSheet(R"(
        QWidget#intrinsicsTabRoot {
            background: #1e1e1e;
        }
        QMainWindow {
            background: #1e1e1e;
        }
        QFrame#brandHeader {
            background: #252526;
            border-bottom: 1px solid #3c3c3c;
        }
        QTabWidget::pane {
            border: 1px solid #3c3c3c;
            background: #1e1e1e;
            top: -1px;
        }
        QTabBar::tab {
            background: #2d2d2d;
            color: #969696;
            padding: 9px 20px;
            border: 1px solid transparent;
            border-bottom: none;
            margin-right: 2px;
        }
        QTabBar::tab:selected {
            background: #1e1e1e;
            color: #ffffff;
            border-top: 2px solid #007acc;
        }
        QTabBar::tab:hover:!selected {
            background: #2a2d2e;
            color: #cccccc;
        }
        QGroupBox {
            background: #252526;
            border: 1px solid #3c3c3c;
            border-radius: 6px;
            margin-top: 16px;
            padding-top: 10px;
            font-weight: 600;
        }
        QGroupBox::title {
            subcontrol-origin: margin;
            left: 10px;
            padding: 0 6px;
            color: #cccccc;
        }
        QLabel {
            color: #cccccc;
        }
        QLabel#infoBanner {
            background: #2d2d2d;
            border: 1px solid #3c3c3c;
            border-radius: 6px;
            padding: 10px 12px;
            color: #9da5b4;
        }
        QLabel#previewArea {
            background: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #141414, stop:1 #1c1c1c);
            border: 1px solid #3c3c3c;
            border-radius: 6px;
            color: #d4d4d4;
        }
        QLineEdit, QComboBox, QSpinBox {
            background: #3c3c3c;
            border: 1px solid #3c3c3c;
            border-radius: 4px;
            padding: 8px 10px;
            min-height: 26px;
            color: #cccccc;
            selection-background-color: #264f78;
        }
        QLineEdit:focus, QComboBox:focus, QSpinBox:focus {
            border: 1px solid #007acc;
        }
        QLineEdit:disabled, QComboBox:disabled, QSpinBox:disabled {
            background: #2d2d2d;
            color: #6b6b6b;
        }
        QComboBox::drop-down {
            border: none;
            width: 22px;
        }
        QComboBox QAbstractItemView {
            background: #3c3c3c;
            color: #cccccc;
            selection-background-color: #094771;
            border: 1px solid #454545;
        }
        QPushButton, QToolButton {
            background: #3c3c3c;
            border: 1px solid #3c3c3c;
            border-radius: 4px;
            padding: 8px 14px;
            min-height: 34px;
            color: #cccccc;
        }
        QPushButton:hover, QToolButton:hover {
            background: #45494e;
        }
        QPushButton:pressed, QToolButton:pressed {
            background: #4d5257;
        }
        QPushButton:disabled, QToolButton:disabled {
            background: #2d2d2d;
            color: #656565;
            border-color: #2d2d2d;
        }
        QPushButton#primaryButton {
            background: #0e639c;
            border: 1px solid #0e639c;
            color: #ffffff;
            font-weight: 600;
        }
        QPushButton#primaryButton:hover {
            background: #1177bb;
        }
        QPushButton#primaryButton:pressed {
            background: #0d5789;
        }
        QPushButton#primaryButton:disabled {
            background: #2d2d2d;
            border-color: #2d2d2d;
            color: #656565;
        }
        QToolButton[toggleState="on"] {
            background: #143d2b;
            border-color: #2ea043;
            color: #89d185;
            font-weight: 700;
        }
        QToolButton[toggleState="off"] {
            background: #3a1d1d;
            border-color: #f14c4c;
            color: #f14c4c;
            font-weight: 700;
        }
        QSlider::groove:horizontal {
            border: 1px solid #3c3c3c;
            height: 6px;
            background: #3c3c3c;
            border-radius: 3px;
        }
        QSlider::handle:horizontal {
            background: #0e639c;
            width: 16px;
            margin: -6px 0;
            border-radius: 8px;
        }
        QSlider::handle:horizontal:hover {
            background: #1177bb;
        }
        QScrollArea#sidebarScroll {
            background: transparent;
            border: none;
        }
        QScrollArea#sidebarScroll > QWidget > QWidget {
            background: transparent;
        }
        QScrollBar:vertical {
            background: transparent;
            width: 12px;
            margin: 0;
        }
        QScrollBar::handle:vertical {
            background: #4a4a4a;
            min-height: 24px;
            border-radius: 5px;
            margin: 2px;
        }
        QScrollBar::handle:vertical:hover {
            background: #5a5a5a;
        }
        QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
            height: 0;
        }
        QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical {
            background: transparent;
        }
    )");

    QMainWindow window;
    window.setWindowTitle("Calibration Tool (Intrinsics + Homography)");

    auto* tabs = new QTabWidget();
    tabs->addTab(new IntrinsicsTab(), "Intrinsics");
    tabs->addTab(new HomographyTab(), "Homography");

    auto* central = new QWidget();
    auto* root_layout = new QVBoxLayout(central);
    root_layout->setContentsMargins(0, 0, 0, 0);
    root_layout->setSpacing(10);
    root_layout->addWidget(make_brand_header());
    root_layout->addWidget(tabs, 1);

    window.setCentralWidget(central);
    window.setMinimumSize(1200, 700);
    window.resize(1280, 760);
    window.show();

    return app.exec();
}
