# C++ Project (WIP)

This folder contains a standalone CMake-based C++ implementation path for the features listed in the repository README.

## Current Scope

- CMake project skeleton with C++17
- Dependency checks aligned to top-level requirements
- Executable stubs for:
  - `calibration_tool_qt` (single UI for intrinsics and homography)
  - `bev_runner`

## Dependency Contract

- OpenCV 4.10.0 or higher
- CMake 3.10 or higher
- Qt5 Widgets
- C++17
- CUDA Toolkit (enabled by default)
- cuDNN (enabled by default)

## OpenCV Resolution

The CMake project prefers OpenCV from `/opt/opencv/4.12.0` by default. An opencv build script is provided in `cpp/assets/build_opencv.sh`.

- Preferred prefix is controlled by `BEV_OPENCV_PREFIX`.
- The build checks `${BEV_OPENCV_PREFIX}/lib/pkgconfig` first.
- If not found, it falls back to system pkg-config OpenCV packages.

Example using a custom prefix:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DBEV_OPENCV_PREFIX=/opt/opencv/4.12.0
```

## CUDA and cuDNN Defaults

CUDA and cuDNN are enabled by default in `CMakeLists.txt`.

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
./build/bev_runner
```

## Mapping to Top-level README Feature Changes

Single calibration UI:
- One Qt application with both workflows in the same tool.
- Intrinsics tab includes left-side fields for chessboard inner corners (x/y) and square size in meters.
- Homography tab includes left-side fields for marker grid width, marker grid height, and camera-to-ground distance.
- Each tab includes a right-side placeholder view finder.

These are scaffolding stubs and do not yet implement full calibration, YAML persistence, undistortion, or BEV warping logic.
