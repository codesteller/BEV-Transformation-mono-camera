#include <QApplication>
#include <QFormLayout>
#include <QGroupBox>
#include <QHBoxLayout>
#include <QLabel>
#include <QLineEdit>
#include <QMainWindow>
#include <QVBoxLayout>
#include <QWidget>

int main(int argc, char** argv) {
    QApplication app(argc, argv);

    QMainWindow window;
    window.setWindowTitle("Extrinsics Calibration (Stub)");

    auto* central = new QWidget();
    auto* root_layout = new QHBoxLayout(central);

    auto* left_panel = new QGroupBox("Marker Geometry Inputs");
    auto* form = new QFormLayout(left_panel);
    form->addRow("Marker Grid Width (m)", new QLineEdit("2.0"));
    form->addRow("Marker Grid Height (m)", new QLineEdit("2.0"));
    form->addRow("Camera to Ground (m)", new QLineEdit("1.5"));

    auto* right_panel = new QGroupBox("View Finder");
    auto* right_layout = new QVBoxLayout(right_panel);
    right_layout->addWidget(new QLabel("Live camera feed with ArUco ID overlays will be rendered here."));

    root_layout->addWidget(left_panel, 1);
    root_layout->addWidget(right_panel, 3);

    window.setCentralWidget(central);
    window.resize(1200, 700);
    window.show();

    return app.exec();
}
