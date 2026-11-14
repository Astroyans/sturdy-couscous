from __future__ import annotations

import threading
import time
from typing import Optional

import cv2
import numpy as np

from src.config import PipelineConfig
from src.models import CameraIntrinsics, FrameBundle

try:
    import pyrealsense2 as rs
except ImportError:  # pragma: no cover
    rs = None


class RealSenseD415Camera:
    def __init__(
        self,
        config: Optional[PipelineConfig] = None,
        use_hardware_filters: bool = False,
    ) -> None:
        self.config = config or PipelineConfig()
        self.use_hardware_filters = use_hardware_filters
        self.pipeline: Optional["rs.pipeline"] = None
        self.intrinsics: Optional[CameraIntrinsics] = None
        self.depth_scale: float = 0.001
        self.device_name: str = "Intel RealSense D415"
        self.usb_type: str = "unknown"
        self.depth_only_mode: bool = False
        self._lock = threading.Lock()
        self._latest_bundle: Optional[FrameBundle] = None
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._align = None

    @staticmethod
    def list_connected_devices() -> list[str]:
        if rs is None:
            return []
        ctx = rs.context()
        return [dev.get_info(rs.camera_info.name) for dev in ctx.query_devices()]

    def _extract_bundle_from_frameset(self, frames) -> Optional[FrameBundle]:
        assert self.intrinsics is not None
        aligned = None
        if self._align is not None and not self.depth_only_mode:
            aligned = self._align.process(frames)
            active_frames = aligned
        else:
            active_frames = frames

        d_frame = active_frames.get_depth_frame()
        if not d_frame:
            if aligned is not None:
                del aligned
            return None

        raw_depth = np.asanyarray(d_frame.get_data()).copy()
        depth_m = raw_depth.astype(np.float32) * self.depth_scale
        h, w = depth_m.shape

        c_frame = active_frames.get_color_frame() if not self.depth_only_mode else None
        if c_frame:
            raw_color = np.asanyarray(c_frame.get_data()).copy()
            if (raw_color.shape[1], raw_color.shape[0]) != (w, h):
                color_bgr = cv2.resize(raw_color, (w, h), interpolation=cv2.INTER_LINEAR)
            else:
                color_bgr = raw_color
        else:
            norm = np.clip((depth_m - 0.25) / 3.0, 0.0, 1.0)
            gray = ((1.0 - norm) * 235.0).astype(np.uint8)
            color_bgr = cv2.applyColorMap(gray, cv2.COLORMAP_BONE)
            color_bgr[depth_m <= 0.05] = (20, 20, 25)

        del c_frame
        del d_frame
        if aligned is not None:
            del aligned

        return FrameBundle(
            timestamp=time.time(),
            color_bgr=color_bgr,
            depth_m=depth_m,
            intrinsics=self.intrinsics,
            imu_yaw=0.0,
            imu_pitch=self.config.nominal_camera_pitch_deg,
            imu_roll=0.0,
        )

    def _capture_loop(self) -> None:
        while self._running and self.pipeline is not None:
            try:
                frames = self.pipeline.poll_for_frames()
                if frames:
                    bundle = self._extract_bundle_from_frameset(frames)
                    del frames
                    if bundle is not None:
                        with self._lock:
                            self._latest_bundle = bundle
                else:
                    time.sleep(0.008)
            except Exception:
                time.sleep(0.02)

    def start(self) -> None:
        if rs is None:
            raise RuntimeError("El paquete 'pyrealsense2' no está instalado.")

        ctx = rs.context()
        devs = ctx.query_devices()
        if len(devs) == 0:
            raise RuntimeError(
                "No se detectó ninguna cámara Intel RealSense conectada por USB."
            )

        dev = devs[0]
        self.device_name = dev.get_info(rs.camera_info.name)
        if dev.supports(rs.camera_info.usb_type_descriptor):
            self.usb_type = dev.get_info(rs.camera_info.usb_type_descriptor)

        # En USB 2.1, desactivar RGB simultaneo evita saturar el bus USB 2.0 y congelar el stream
        if self.usb_type.startswith("2"):
            candidate_profiles = [
                ((480, 270, 6), None),
                ((640, 480, 6), None),
                ((256, 144, 90), None),
            ]
        else:
            candidate_profiles = [
                (
                    (self.config.width, self.config.height, self.config.fps),
                    (self.config.width, self.config.height, self.config.fps),
                ),
                ((640, 480, 30), (640, 480, 30)),
                ((640, 480, 15), (640, 480, 15)),
                ((480, 270, 6), None),
            ]

        last_err: Optional[Exception] = None
        active_profile = None
        first_bundle: Optional[FrameBundle] = None

        for depth_spec, color_spec in candidate_profiles:
            pipeline = None
            try:
                dw, dh, dfps = depth_spec
                pipeline = rs.pipeline()
                rs_cfg = rs.config()
                rs_cfg.enable_stream(rs.stream.depth, dw, dh, rs.format.z16, dfps)
                if color_spec is not None:
                    cw, ch, cfps = color_spec
                    rs_cfg.enable_stream(rs.stream.color, cw, ch, rs.format.bgr8, cfps)

                profile = pipeline.start(rs_cfg)
                depth_sensor = profile.get_device().first_depth_sensor()
                self.depth_scale = float(depth_sensor.get_depth_scale())

                if self.usb_type.startswith("2"):
                    if depth_sensor.supports(rs.option.laser_power):
                        depth_sensor.set_option(rs.option.laser_power, 60.0)
                    if depth_sensor.supports(rs.option.global_time_enabled):
                        depth_sensor.set_option(rs.option.global_time_enabled, 0.0)
                else:
                    if depth_sensor.supports(rs.option.emitter_enabled):
                        depth_sensor.set_option(rs.option.emitter_enabled, 1.0)
                    if depth_sensor.supports(rs.option.laser_power):
                        lp_range = depth_sensor.get_option_range(rs.option.laser_power)
                        depth_sensor.set_option(rs.option.laser_power, float(lp_range.max))

                self.depth_only_mode = color_spec is None
                self._align = None if self.depth_only_mode else rs.align(rs.stream.color)

                stream_type = rs.stream.depth if self.depth_only_mode else rs.stream.color
                target_stream = profile.get_stream(stream_type).as_video_stream_profile()
                intr = target_stream.get_intrinsics()
                self.intrinsics = CameraIntrinsics(
                    width=int(intr.width),
                    height=int(intr.height),
                    fx=float(intr.fx),
                    fy=float(intr.fy),
                    ppx=float(intr.ppx),
                    ppy=float(intr.ppy),
                    depth_scale=self.depth_scale,
                )

                frames = pipeline.wait_for_frames(timeout_ms=4000)
                first_bundle = self._extract_bundle_from_frameset(frames)
                del frames
                if first_bundle is None:
                    raise RuntimeError("No llegó frame inicial de profundidad.")

                self.pipeline = pipeline
                active_profile = profile
                break
            except Exception as exc:
                last_err = exc
                if pipeline is not None:
                    try:
                        pipeline.stop()
                    except Exception:
                        pass
                continue

        if active_profile is None or self.pipeline is None or first_bundle is None:
            raise RuntimeError(
                f"No fue posible iniciar el stream de {self.device_name} (USB {self.usb_type}): {last_err}"
            )

        with self._lock:
            self._latest_bundle = first_bundle

        self._running = True
        self._thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._thread.start()

    def read_frame(self, timeout_ms: int = 1500) -> FrameBundle:
        _ = timeout_ms
        with self._lock:
            bundle = self._latest_bundle
        if bundle is None:
            raise RuntimeError("La cámara RealSense D415 no tiene frames disponibles.")
        return FrameBundle(
            timestamp=time.time(),
            color_bgr=bundle.color_bgr.copy(),
            depth_m=bundle.depth_m.copy(),
            intrinsics=bundle.intrinsics,
            imu_yaw=bundle.imu_yaw,
            imu_pitch=bundle.imu_pitch,
            imu_roll=bundle.imu_roll,
        )

    def stop(self) -> None:
        self._running = False
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=1.5)
        self._thread = None
        if self.pipeline is not None:
            try:
                self.pipeline.stop()
            except Exception:
                pass
            self.pipeline = None

    def __enter__(self) -> RealSenseD415Camera:
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.stop()
