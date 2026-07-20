# Monocular Calibration Tool with BEV Transformation & Depth Estimation
It converts a video input to BEV representation for depth estimation and object detection. The repository can be used for the entire calibration process of monocular camera to BEV representation.

## Implementation

Active development is in [`cpp/`](cpp/) — a Qt5 + OpenCV desktop calibration tool. See [cpp/README.md](cpp/README.md) for build/run instructions and full feature details. [`python/`](python/) holds an earlier, unmaintained reference implementation of the same calibration steps.

## Features
- [x] Monocular camera intrinsics calibration based on OpenCV
    - [x] Qt UI with a live view finder, chessboard detection overlay, and ROS-style coverage (X/Y/Size/Skew) tracking to guide capture, matching the OpenCV/ROS calibration tool workflow.
    - [x] Left-side fields for chessboard inner corners (rows/columns) and square size in meters, plus camera and resolution selection.
    - [x] Save the calibration result (camera matrix, distortion, rectification, projection) to a ROS-CameraInfo-style YAML file, alongside the captured calibration frames.
    - [x] Once calibrated, intrinsics are applied automatically to undistort the image anywhere they're loaded (the homography tab's preview).
- [x] Monocular camera extrinsics calibration based on OpenCV 4-point homography
    - [x] Same Qt window, second tab, with a view finder supporting manual 4-point clicking or automatic detection via 4 ArUco markers on the ground plane.
    - [x] Left-side fields for the real-world plane width/height in meters. Camera height and ground position are recovered automatically via `solvePnP` from those dimensions plus the 4 point correspondences — no manual tape measurement to each corner required.
    - [x] Save the homography, recovered camera pose, and the intrinsics it was solved against to a single YAML file.
    - [x] `bev_runner` loads the saved intrinsics + homography YAML, undistorts live frames, and warps them to a top-down BEV view with the saved homography (plus a scale grid and camera-position marker for reference).
    - [x] Use the BEV transformation for depth estimation and object detection — the Homography tab's drag-a-bounding-box validation mode maps any point to ground-plane distance from the camera.

## Dependencies
- OpenCV 4.10.0 or higher
- CMake 3.10 or higher
- Qt5
- C++17
- CUDA Toolkit / cuDNN (optional today; for future GPU-accelerated depth estimation and object detection)
