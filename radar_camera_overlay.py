#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Copyright: Copyright (C) 2026 Gahan Ai Pvt Ltd
# @Description: Standalone live sanity-check for a solved radar->camera extrinsic - no Qt, no
#              calibration_tool_qt, no intrinsics/homography wiring beyond reading the two YAML
#              files needed for the math. Projects every raw radar detection onto the live camera
#              feed using the extrinsic in radar_extrinsics.yaml, so you can eyeball alignment
#              without opening the full calibration tool.
"""
Radar CAN-FD decode is imported directly from radar_4d_cli.py (RadarCANFDParser/RadarState/
dispatch) rather than reimplemented here - that module is itself standalone (no ROS/Qt), see its
own docstring. Keep decode logic in exactly one place; re-sync manually if it ever changes.

Usage:
    python3 radar_camera_overlay.py \\
        --intrinsics ~/.calibration/openadas/cam0/intrinsics.yaml \\
        --extrinsics ~/.calibration/openadas/cam0/radar_extrinsics.yaml \\
        --camera-index 2 --can-channel can1

Press 'q' or Esc in the window to quit.
"""
from __future__ import annotations

import argparse
import sys
import threading
from typing import Optional, Tuple

import cv2
import numpy as np
import yaml

import radar_4d_cli as radar_cli


def load_intrinsics(path: str) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray], Tuple[int, int]]:
    with open(path) as f:
        data = yaml.safe_load(f)
    camera_matrix = np.array(data["camera_matrix"]["data"], dtype=np.float64).reshape(3, 3)
    dist_coeffs = np.array(data["distortion_coefficients"]["data"], dtype=np.float64)
    projection_matrix = None
    if "projection_matrix" in data:
        # ROS-style 3x4 P -- only the top-left 3x3 is a usable camera matrix (last column is the
        # stereo baseline term, always 0 for a monocular camera), matching
        # bev::calibration_io.hpp's load_intrinsics_yaml convention.
        p_full = np.array(data["projection_matrix"]["data"], dtype=np.float64).reshape(3, 4)
        projection_matrix = p_full[:, :3]
    width = int(data["image_width"])
    height = int(data["image_height"])
    return camera_matrix, dist_coeffs, projection_matrix, (width, height)


def load_extrinsics(path: str) -> Tuple[np.ndarray, np.ndarray]:
    with open(path) as f:
        data = yaml.safe_load(f)
    R = np.array(data["R"], dtype=np.float64).reshape(3, 3)
    t = np.array(data["t"], dtype=np.float64).reshape(3)
    return R, t


def effective_camera_matrix(camera_matrix: np.ndarray, projection_matrix: Optional[np.ndarray]) -> np.ndarray:
    # Rectify-before-fusion convention shared with radar_cam_tab.hpp::effective_camera_matrix(): use
    # the projection matrix (already accounts for rectification) when present, else the raw K.
    return projection_matrix if projection_matrix is not None else camera_matrix


def project_pinhole(K: np.ndarray, p_camera: np.ndarray) -> Optional[Tuple[float, float]]:
    if p_camera[2] <= 0.05:  # behind or at the camera -- not projectable
        return None
    x = p_camera[0] / p_camera[2]
    y = p_camera[1] / p_camera[2]
    return K[0, 0] * x + K[0, 2], K[1, 1] * y + K[1, 2]


def radar_reader_loop(state: "radar_cli.RadarState", parser: "radar_cli.RadarCANFDParser", bus,
                       stop_event: threading.Event) -> None:
    while not stop_event.is_set():
        try:
            msg = bus.recv(timeout=0.5)
        except Exception as exc:
            print(f"[warn] CAN recv error: {exc}", file=sys.stderr)
            continue
        if msg is None:
            continue
        radar_cli.dispatch(state, parser, msg.arbitration_id, bytes(msg.data))


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--intrinsics", required=True, help="Path to intrinsics.yaml")
    p.add_argument("--extrinsics", required=True, help="Path to radar_extrinsics.yaml")
    p.add_argument("--camera-index", type=int, default=0)
    p.add_argument("--can-channel", default="can1")
    p.add_argument("--bitrate", type=int, default=500000)
    p.add_argument("--data-bitrate", type=int, default=2000000)
    p.add_argument("--point-radius", type=int, default=5)
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)

    camera_matrix, dist_coeffs, projection_matrix, (width, height) = load_intrinsics(args.intrinsics)
    K = effective_camera_matrix(camera_matrix, projection_matrix)
    R, t = load_extrinsics(args.extrinsics)
    print(f"[info] loaded intrinsics ({width}x{height}) from {args.intrinsics}")
    print(f"[info] loaded extrinsics from {args.extrinsics}")

    cap = cv2.VideoCapture(args.camera_index, cv2.CAP_V4L2)
    if not cap.isOpened():
        cap = cv2.VideoCapture(args.camera_index)
    if not cap.isOpened():
        print(f"[FATAL] could not open camera {args.camera_index}", file=sys.stderr)
        return 1
    # MJPG is required to unlock the intrinsics' capture resolution on most UVC cameras, same fix
    # as calibration_tool_qt's preview start -- without it V4L2 silently falls back to a low default.
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    try:
        bus = radar_cli.can.interface.Bus(
            channel=args.can_channel, interface="socketcan", fd=True, bitrate=args.bitrate,
        )
    except Exception as exc:
        print(f"[FATAL] could not open CAN channel {args.can_channel}: {exc}", file=sys.stderr)
        cap.release()
        return 1

    state = radar_cli.RadarState()
    parser = radar_cli.RadarCANFDParser()
    stop_event = threading.Event()
    reader = threading.Thread(target=radar_reader_loop, args=(state, parser, bus, stop_event), daemon=True)
    reader.start()

    print("[info] running -- press 'q' or Esc in the window to quit")
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                continue
            undistorted = cv2.undistort(frame, camera_matrix, dist_coeffs, None, K)

            with state.lock:
                detections = list(state.detections)

            for d in detections:
                q = np.array([d.pos_x, d.pos_y, d.pos_z])
                p_cam = R @ q + t
                px = project_pinhole(K, p_cam)
                if px is None:
                    continue
                x, y = int(px[0]), int(px[1])
                if 0 <= x < undistorted.shape[1] and 0 <= y < undistorted.shape[0]:
                    cv2.circle(undistorted, (x, y), args.point_radius, (217, 89, 38), 2, cv2.LINE_AA)

            cv2.putText(undistorted, f"detections: {len(detections)}", (10, 24),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (210, 210, 210), 1, cv2.LINE_AA)
            cv2.imshow("Radar-Camera Overlay", undistorted)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
    finally:
        stop_event.set()
        reader.join(timeout=1.0)
        bus.shutdown()
        cap.release()
        cv2.destroyAllWindows()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
