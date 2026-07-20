# C++ Calibration Tool

Standalone CMake-based C++ implementation of the calibration workflow described in the
[repository README](../README.md): a Qt5 + OpenCV desktop app that takes a monocular camera from
raw feed to a saved intrinsics + ground-plane homography, ready for a downstream BEV pipeline.

## Executables

- **`calibration_tool_qt`** — the actively developed tool. A single window with two tabs:
  - **Intrinsics**: camera + resolution selection (probes actual supported modes over MJPG so
    cameras aren't silently capped at a low default resolution), live preview with chessboard
    detection and ROS-style X/Y/Size/Skew coverage tracking, capture, and
    `cv::calibrateCamera` + save to YAML.
  - **Homography**: loads a saved intrinsics YAML and rectifies the live preview with it, then
    solves a ground-plane homography from 4 points — clicked manually, or auto-detected via 4
    ArUco markers. Camera position and height are recovered automatically with `cv::solvePnP`
    from the plane's known real-world size (width/height fields) and the 4 point
    correspondences, so no manual distance measurements to each corner are needed. A zoom loupe
    (press `Z` on the preview, click inside the magnified inset) helps place corner points
    precisely. A drag-a-bounding-box validation mode reports estimated distance-to-camera for
    sanity-checking against a tape measurement.
- **`bev_runner`** — loads a saved intrinsics + homography YAML pair, opens the camera at the
  intrinsics' resolution, and per frame: undistorts, then warps straight to a top-down BEV
  canvas by composing the saved homography with a meters-to-pixels transform. The view window
  auto-sizes to always include both the calibrated plane and the camera's own recovered ground
  position, with a 1m reference grid, the plane's footprint outline, and a camera-position
  marker drawn on top. Displays "Rectified" and "BEV" windows live (`q`/Esc to quit); see
  `./build/bev_runner --help` for resolution/margin flags.
- `intrinsics_calibrator_qt`, `extrinsics_calibrator_qt` — early standalone-window stubs
  superseded by the unified `calibration_tool_qt`. Not part of the CMake build.

Shared, Qt-free YAML readers for both executables live in
[`include/bev/calibration_io.hpp`](include/bev/calibration_io.hpp), so `calibration_tool_qt` and
`bev_runner` can never disagree about the file format.

## Default Asset Location

Calibration outputs default to `${HOME}/.calibration/openadas/cam<id>/`, where `<id>` is the
camera's device index (e.g. `/dev/video2` → `cam2`). This keeps multiple cameras' assets from
overwriting each other, and the path updates live as you switch the Camera dropdown:

```
${HOME}/.calibration/openadas/cam<id>/intrinsics.yaml
${HOME}/.calibration/openadas/cam<id>/intrinsics_images/   # saved calibration frames
${HOME}/.calibration/openadas/cam<id>/homography.yaml
```

Every path field has a Browse button if you want a different location instead.

## Dependency Contract

- OpenCV 4.10.0 or higher
- CMake 3.10 or higher
- Qt5 Widgets
- C++17
- CUDA Toolkit (enabled by default)
- cuDNN (enabled by default)

## OpenCV Resolution

The CMake project prefers OpenCV from `/opt/opencv/4.12.0` by default. An OpenCV build script is
provided in `cpp/assets/build_opencv.sh`.

- Preferred prefix is controlled by `BEV_OPENCV_PREFIX`.
- The build checks `${BEV_OPENCV_PREFIX}/lib/pkgconfig` first.
- If not found, it falls back to system pkg-config OpenCV packages.

Example using a custom prefix:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DBEV_OPENCV_PREFIX=/opt/opencv/4.12.0
```

## CUDA and cuDNN Defaults

CUDA and cuDNN are enabled by default in `CMakeLists.txt`. Neither is used by the calibration
tool itself yet — they're checked/linked ahead of the planned `bev_runner` GPU work.

- `ENABLE_CUDA=ON` (default)
- `ENABLE_CUDNN=ON` (default)

Disable explicitly if needed:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DENABLE_CUDA=OFF -DENABLE_CUDNN=OFF
```

## Build

```bash
cd cpp
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

Example build with explicit defaults shown:

```bash
cd cpp
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DBEV_OPENCV_PREFIX=/opt/opencv/4.12.0 -DENABLE_CUDA=ON -DENABLE_CUDNN=ON
cmake --build build -j
```

## Run

```bash
./build/calibration_tool_qt
./build/bev_runner [camera_index] [--intrinsics PATH] [--homography PATH] [--margin-m N] [--px-per-m N]
```

`bev_runner` defaults `camera_index` to `0` and both YAML paths to
`${HOME}/.calibration/openadas/cam<camera_index>/{intrinsics,homography}.yaml` — the same
default location the GUI tool saves to, so a plain `./build/bev_runner` picks up whatever you
just calibrated for camera 0.

## Typical Workflow

1. **Intrinsics tab**: pick the camera, scan/select its working resolution, set the chessboard
   inner-corner counts and square size, start preview, capture chessboard frames until coverage
   looks reasonable, then Calibrate + Save. This writes `cam<id>/intrinsics.yaml` and the
   captured frames.
2. **Homography tab**: pick the same camera, load that `intrinsics.yaml` (auto-filled by default),
   enter the ground-plane target's real-world width/height, start preview, and pick the 4 target
   corners (manual click or ArUco auto-detect) in TL → TR → BR → BL order. Use the zoom loupe
   (`Z`) for precise clicks. Solve + Save writes `cam<id>/homography.yaml`, including the
   solved homography, the `solvePnP`-recovered camera pose, and the intrinsics it was solved
   against.
3. Optionally use Validation mode to drag a box around a real object and compare the tool's
   estimated distance against a tape measurement.
4. Run `./build/bev_runner` (or `bev_runner <camera_index>` for a different camera) to see the
   live undistorted feed and its BEV warp side by side, using the same two YAML files.
