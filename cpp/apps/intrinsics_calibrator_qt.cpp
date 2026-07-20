#include <QApplication>
#include <QFormLayout>
#include <QGroupBox>
#include <QHBoxLayout>
#include <QLabel>
#include <QLineEdit>
#include <QMainWindow>
#include <QSplitter>
#include <QVBoxLayout>
#include <QWidget>

int main(int argc, char** argv) {
    QApplication app(argc, argv);

    QMainWindow window;
    window.setWindowTitle("Intrinsics Calibration (Stub)");

    auto* central = new QWidget();
    auto* root_layout = new QHBoxLayout(central);

    auto* left_panel = new QGroupBox("Calibration Inputs");
    auto* form = new QFormLayout(left_panel);
    form->addRow("Chessboard Inner Corners X", new QLineEdit("8"));
    form->addRow("Chessboard Inner Corners Y", new QLineEdit("6"));
    form->addRow("Square Size (m)", new QLineEdit("0.035"));

    auto* right_panel = new QGroupBox("View Finder");
    auto* right_layout = new QVBoxLayout(right_panel);
    right_layout->addWidget(new QLabel("Live camera feed and chessboard detection overlay will be rendered here."));

    root_layout->addWidget(left_panel, 1);
    root_layout->addWidget(right_panel, 3);

    window.setCentralWidget(central);
    window.resize(1200, 700);
    window.show();

    return app.exec();
}
