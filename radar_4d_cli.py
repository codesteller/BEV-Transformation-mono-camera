#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Copyright: Copyright (C) 2026 Gahan Ai Pvt Ltd
# @Author: Pavan Patil
# @Date: 2026-07-20
# @Last Modified by:   Pavan Patil
# @Last Modified time: 2026-07-20
# @Description: Standalone CAN-FD parser + CLI printer for the Columbus FC 4D radar - no
#              ROS, no drwig_lrr_pro_stack dependency, just python-can.
"""
Fully standalone - not a ROS driver, not part of drwig_lrr_pro_stack's git
repo. RX only, matching the radar's own behavior (it has no vehicle-input
TX path, unlike the DRWIG LRR Pro/ctlrr220pro radar).

Decode logic (RDI/OD bit-math) is copied verbatim from
/home/asus/Developer/open_adas_ros/src/drwig_4D_radar/drwig_4d_radar/4D_radar_node.py's
RadarCANFDParser + Detection/TrackedObject dataclasses - that class has no
ROS dependency itself (only the file it lives in does, at import time), so
it's copied here rather than imported, to keep this script runnable
without a ROS2 environment. Re-sync manually if that upstream file's
decode logic changes - there is no shared package linking the two.

CAN-FD IDs:
    0x281/0x282/0x283/0x284 - RDI (raw/detected point cloud)
    0x381/0x382             - OD  (tracked objects)

Usage:
    python3 radar_4d_cli.py
    python3 radar_4d_cli.py --channel can1 --bitrate 500000 --data-bitrate 2000000
    python3 radar_4d_cli.py --print-interval-s 0.5 --show-detections
"""
from __future__ import annotations

import argparse
import math
import signal
import sys
import threading
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

try:
    import can
except ImportError:
    print("[ERROR] python-can not installed. Run: pip install python-can", file=sys.stderr)
    sys.exit(1)

g_stop = threading.Event()


def handle_signal(_sig, _frame) -> None:
    g_stop.set()


# ══════════════════════════════════════════════════════════════════════════
# Decode - copied from 4D_radar_node.py's RadarCANFDParser, unchanged.
# ══════════════════════════════════════════════════════════════════════════


@dataclass
class Detection:
    range_m: float = 0.0
    vel_mps: float = 0.0
    azimuth_rad: float = 0.0
    elevation_rad: float = 0.0
    rcs_dbsm: float = 0.0
    snr_db: float = 0.0
    is_stationary: bool = False
    pos_x: float = 0.0
    pos_y: float = 0.0
    pos_z: float = 0.0


@dataclass
class TrackedObject:
    track_id: int = 0
    pos_x: float = 0.0
    pos_y: float = 0.0
    pos_z: float = 0.0
    vel_x: float = 0.0
    vel_y: float = 0.0
    length: float = 0.0
    width: float = 0.0
    height: float = 0.0
    orientation: float = 0.0
    exist_prob: float = 0.0
    obstacle_prob: float = 0.0
    obj_class: int = 0
    rcs_dbsm: float = 0.0
    age: int = 0
    timestamp: float = 0.0


class RadarCANFDParser:
    """Decodes RDI (0x281-0x284) and OD (0x381-0x382) CAN-FD frames."""

    @staticmethod
    def _s16(v: int) -> int:
        return v - 0x10000 if v & 0x8000 else v

    def parse_rdi_header(self, data: bytes) -> dict:
        if len(data) < 13:
            return {}
        return dict(
            radar_id=(data[10] >> 2) & 0x07,
            frame_id=(data[8] << 8) | data[9],
            timestamp=(data[4] << 24) | (data[5] << 16) | (data[6] << 8) | data[7],
            det_number=((data[11] & 0x7F) << 3) | ((data[12] & 0xE0) >> 5),
        )

    def parse_rdi_slot(self, p: bytes) -> Optional[Detection]:
        if len(p) < 19:
            return None
        d = Detection()

        raw_r = ((p[2] & 0x3F) << 9) | (p[3] << 1) | ((p[4] >> 7) & 1)
        d.range_m = raw_r * 0.01

        raw_v = ((p[5] & 0x07) << 12) | (p[6] << 4) | ((p[7] >> 4) & 0xF)
        d.vel_mps = raw_v * 0.01 - 83.3

        raw_az = ((p[8] & 1) << 15) | (p[9] << 7) | ((p[10] >> 1) & 0x7F)
        d.azimuth_rad = self._s16(raw_az) * 0.000175 - 3.15

        raw_el = ((p[12] & 7) << 13) | (p[13] << 5) | ((p[14] >> 3) & 0x1F)
        d.elevation_rad = self._s16(raw_el) * 0.000175 - 3.15

        d.rcs_dbsm = float((p[16] & 0x1F) * 2 - 40)
        d.snr_db = ((p[17] & 0xFE) >> 1) * 0.5
        d.is_stationary = bool((p[17] & 0x01) ^ 0x01)

        az = d.azimuth_rad
        el = d.elevation_rad
        cos_el = math.cos(el)
        d.pos_x = d.range_m * cos_el * math.cos(az)
        d.pos_y = d.range_m * cos_el * math.sin(az)
        d.pos_z = d.range_m * math.sin(el)
        return d

    def parse_rdi_body(self, data: bytes) -> List[Detection]:
        slot = 19
        return [
            d
            for i in range(len(data) // slot)
            if (d := self.parse_rdi_slot(data[i * slot : (i + 1) * slot])) is not None
        ]

    def parse_od_header(self, data: bytes) -> dict:
        if len(data) < 6:
            return {}
        return dict(frame_id=(data[1] << 8) | data[2], obj_count=data[5])

    def parse_od_slot(self, p: bytes) -> Optional[TrackedObject]:
        if len(p) < 44:
            return None
        o = TrackedObject()
        o.track_id = p[1]

        px_raw = (p[2] << 7) | (p[3] >> 1)
        py_raw = ((p[3] & 1) << 14) | (p[4] << 6) | (p[5] >> 2)
        o.pos_x = px_raw * 0.025 - 400
        o.pos_y = py_raw * 0.025 - 400

        vx_raw = ((p[9] & 0xF) << 10) | (p[10] << 2) | (p[11] >> 6)
        vy_raw = ((p[11] & 0x3F) << 8) | p[12]
        vx_raw = self._s16(vx_raw << 4) >> 4
        vy_raw = self._s16(vy_raw << 4) >> 4
        o.vel_x = vx_raw * 0.02
        o.vel_y = vy_raw * 0.02

        o.length = ((p[13] << 4) | (p[14] >> 4)) * 0.1
        o.width = (((p[14] & 0xF) << 8) | p[15]) * 0.1
        o.height = ((p[16] << 4) | (p[17] >> 4)) * 0.1

        o.orientation = (((p[17] & 0xF) << 8) | p[18]) * 0.0175 - 4.48
        exist_raw = p[19] >> 1
        obstacle_raw = ((p[19] & 1) << 6) | (p[20] >> 2)
        o.exist_prob = (exist_raw * 2) / 100.0
        o.obstacle_prob = (obstacle_raw * 2) / 100.0
        o.obj_class = ((p[20] & 0x3) << 2) | (p[21] >> 6)
        o.rcs_dbsm = (p[21] & 0x3F) * 0.5 - 15
        o.age = p[22]
        o.pos_z = o.height / 2.0
        o.timestamp = time.time()
        return o

    def parse_od_body(self, data: bytes) -> List[TrackedObject]:
        slot = 44
        return [
            o
            for i in range(len(data) // slot)
            if (o := self.parse_od_slot(data[i * slot : (i + 1) * slot])) is not None
        ]


# ══════════════════════════════════════════════════════════════════════════
# State + dispatch - same shape as 4D_radar_node.py's RadarState/_dispatch,
# minus everything ROS-specific (no publishers, no TF, no markers).
# ══════════════════════════════════════════════════════════════════════════


@dataclass
class RadarState:
    detections: List[Detection] = field(default_factory=list)
    tracks: Dict[int, TrackedObject] = field(default_factory=dict)
    frame_id: int = 0
    radar_id: int = 0
    total_msgs: int = 0
    total_dets: int = 0
    total_tracks: int = 0
    errors: int = 0
    fd_frames: int = 0
    lock: threading.Lock = field(default_factory=threading.Lock)
    _pending: List[Detection] = field(default_factory=list)
    _expected_det_frames: int = 1
    _received_det_frames: int = 0


def dispatch(state: RadarState, parser: RadarCANFDParser, cid: int, data: bytes) -> None:
    with state.lock:
        state.total_msgs += 1

    if cid == 0x281:
        hdr = parser.parse_rdi_header(data)
        if hdr:
            with state.lock:
                state.frame_id = hdr["frame_id"]
                state.radar_id = hdr["radar_id"]
                state._pending = []
                n = hdr["det_number"]
                state._expected_det_frames = (n // 3) + (1 if n % 3 else 0) or 1
                state._received_det_frames = 0

    elif cid in (0x282, 0x283, 0x284):
        dets = parser.parse_rdi_body(data)
        with state.lock:
            state._pending.extend(dets)
            state.total_dets += len(dets)
            state._received_det_frames += 1
            if state._received_det_frames >= state._expected_det_frames:
                state.detections = list(state._pending)

    elif cid == 0x381:
        parser.parse_od_header(data)

    elif cid == 0x382:
        objs = parser.parse_od_body(data)
        now = time.time()
        with state.lock:
            for o in objs:
                state.tracks[o.track_id] = o
                state.total_tracks += 1
            state.tracks = {k: v for k, v in state.tracks.items() if now - v.timestamp < 2.0}


# ══════════════════════════════════════════════════════════════════════════
# CLI print loop
# ══════════════════════════════════════════════════════════════════════════


def print_snapshot(state: RadarState, show_detections: bool, show_tracks: bool) -> None:
    with state.lock:
        detections = list(state.detections)
        tracks = list(state.tracks.values())
        frame_id = state.frame_id
        radar_id = state.radar_id
        total_msgs = state.total_msgs
        total_dets = state.total_dets
        total_tracks = state.total_tracks
        errors = state.errors
        fd_frames = state.fd_frames

    print(
        f"\n--- 4D radar snapshot --- frame={frame_id} radar_id={radar_id} "
        f"detections={len(detections)} tracks={len(tracks)} total_msgs={total_msgs} "
        f"fd_frames={fd_frames} errors={errors} (cumulative dets={total_dets} "
        f"tracks={total_tracks})"
    )

    if show_detections:
        for d in detections:
            print(
                f"  det  r={d.range_m:6.1f}m  az={math.degrees(d.azimuth_rad):6.1f}deg  "
                f"v={d.vel_mps:6.2f}m/s  rcs={d.rcs_dbsm:5.1f}dBsm  "
                f"stationary={d.is_stationary}"
            )

    if show_tracks:
        for t in tracks:
            speed = math.sqrt(t.vel_x**2 + t.vel_y**2)
            print(
                f"  trk  id={t.track_id:3d}  x={t.pos_x:6.1f}m  y={t.pos_y:6.1f}m  "
                f"v={speed:5.1f}m/s  W={t.width:.1f} L={t.length:.1f} H={t.height:.1f}m  "
                f"class={t.obj_class}  age={t.age}  exist={t.exist_prob:.0%}"
            )


def run(channel: str, bitrate: int, data_bitrate: int, print_interval_s: float,
        show_detections: bool, show_tracks: bool) -> int:
    state = RadarState()
    parser = RadarCANFDParser()

    try:
        bus = can.interface.Bus(
            channel=channel, interface="socketcan", fd=True, bitrate=bitrate,
        )
        print(f"[info] opened {channel} (CAN-FD, bitrate={bitrate}, data_bitrate={data_bitrate})")
    except Exception as exc:
        print(f"[FATAL] could not open {channel}: {exc}", file=sys.stderr)
        return 1

    print("[info] listening - Ctrl+C to stop")
    last_print = 0.0
    while not g_stop.is_set():
        try:
            msg = bus.recv(timeout=0.5)
        except Exception as exc:
            print(f"[warn] recv error: {exc}", file=sys.stderr)
            with state.lock:
                state.errors += 1
            continue
        if msg is None:
            continue

        cid = msg.arbitration_id
        data = bytes(msg.data)
        if getattr(msg, "is_fd", False):
            with state.lock:
                state.fd_frames += 1
        dispatch(state, parser, cid, data)

        now = time.time()
        if now - last_print >= print_interval_s:
            last_print = now
            print_snapshot(state, show_detections, show_tracks)

    bus.shutdown()
    print("\n[info] stopped")
    return 0


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--channel", default="can1", help="SocketCAN interface (default: can1)")
    p.add_argument("--bitrate", type=int, default=500000, help="CAN arbitration bitrate")
    p.add_argument("--data-bitrate", type=int, default=2000000, help="CAN-FD data bitrate")
    p.add_argument("--print-interval-s", type=float, default=1.0, help="Console print cadence")
    p.add_argument("--show-detections", action="store_true", help="Also print each raw detection")
    p.add_argument("--show-tracks", action="store_true", default=True,
                    help="Print tracked objects (default: on)")
    p.add_argument("--no-show-tracks", dest="show_tracks", action="store_false")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)
    return run(
        args.channel, args.bitrate, args.data_bitrate, args.print_interval_s,
        args.show_detections, args.show_tracks,
    )


if __name__ == "__main__":
    raise SystemExit(main())
