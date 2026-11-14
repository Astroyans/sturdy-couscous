from __future__ import annotations

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
        use_hardware_filters: bool = True,
    ) -> None:
        self.config = config or PipelineConfig()
        self.use_hardware_filters = use_hardware_filters
        self.pipeline: Optional["rs.pipeline"] = None
        self.align: Optional["rs.align"] = None
        self.intrinsics: Optional[CameraIntrinsics] = None
        self.depth_scale: float = 0.001
        self.device_name: str = "Intel RealSense D415"
        self.usb_type: str = "unknown"
        self.depth_only_mode: bool = False
        self._active_spec: Optional[tuple] = None
        self._last_bundle: Optional[FrameBundle] = None
        self._spatial_filter = None
        self._temporal_filter = None
        self._hole_filling_filter = None

    @staticmethod
    def list_connected_devices() -> list[str]:
        if rs is None:
            return []
        ctx = rs.context()
        return [dev.get_info(rs.camera_info.name) for dev in ctx.query_devices()]

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

        if self.usb_type.startswith("2"):
            candidate_profiles = [
                ((480, 270, 6), (424, 240, 6)),
                ((480, 270, 6), None),
                ((640, 480, 6), None),
            ]
        else:
            candidate_profiles = [
                (
                    (self.config.width, self.config.height, self.config.fps),
                    (self.config.width, self.config.height, self.config.fps),
                ),
                ((640, 480, 30), (640, 480, 30)),
                ((640, 480, 15), (640, 480, 15)),
                ((480, 270, 6), (424, 240, 6)),
                ((480, 270, 6), None),
            ]

        last_err: Optional[Exception] = None
        active_profile = None
        initial_frames = None

        for depth_spec, color_spec in candidate_profiles:
            dw, dh, dfps = depth_spec
            pipeline = rs.pipeline()
            rs_cfg = rs.config()
            rs_cfg.enable_stream(rs.stream.depth, dw, dh, rs.format.z16, dfps)
            if color_spec is not None:
                cw, ch, cfps = color_spec
                rs_cfg.enable_stream(rs.stream.color, cw, ch, rs.format.bgr8, cfps)

            try:
                profile = pipeline.start(rs_cfg)
                frames = pipeline.wait_for_frames(timeout_ms=4500)
                if not frames.get_depth_frame():
                    raise RuntimeError("No llegó frame de profundidad.")
                self.pipeline = pipeline
                self.depth_only_mode = color_spec is None
                self._active_spec = (depth_spec, color_spec)
                active_profile = profile
                initial_frames = frames
                break
            except Exception as exc:
                last_err = exc
                try:
                    pipeline.stop()
                except Exception:
                    pass
                continue

        if active_profile is None or self.pipeline is None:
            raise RuntimeError(
                f"No fue posible iniciar el stream de {self.device_name} (USB {self.usb_type}): {last_err}"
            )

        depth_sensor = active_profile.get_device().first_depth_sensor()
        self.depth_scale = float(depth_sensor.get_depth_scale())

        if not self.depth_only_mode:
            self.align = rs.align(rs.stream.color)
            target_stream = active_profile.get_stream(
                rs.stream.color
            ).as_video_stream_profile()
        else:
            self.align = None
            target_stream = active_profile.get_stream(
                rs.stream.depth
            ).as_video_stream_profile()

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

        if self.use_hardware_filters:
            self._spatial_filter = rs.spatial_filter()
            self._spatial_filter.set_option(rs.option.filter_magnitude, 2)
            self._spatial_filter.set_option(rs.option.filter_smooth_alpha, 0.5)
            self._spatial_filter.set_option(rs.option.filter_smooth_delta, 20)

            self._temporal_filter = rs.temporal_filter()
            self._temporal_filter.set_option(rs.option.filter_smooth_alpha, 0.4)
            self._temporal_filter.set_option(rs.option.filter_smooth_delta, 20)

            self._hole_filling_filter = rs.hole_filling_filter(1)

        if initial_frames is not None:
            self._last_bundle = self._convert_frames(initial_frames)

    def _convert_frames(self, frames) -> FrameBundle:
        assert self.intrinsics is not None
        if self.align is not None:
            frames = self.align.process(frames)

        depth_frame = frames.get_depth_frame()
        if self.use_hardware_filters:
            if self._spatial_filter is not None:
                depth_frame = self._spatial_filter.process(depth_frame)
            if self._temporal_filter is not None:
                depth_frame = self._temporal_filter.process(depth_frame)
            if self._hole_filling_filter is not None:
                depth_frame = self._hole_filling_filter.process(depth_frame)

        raw_depth = np.asanyarray(depth_frame.get_data())
        depth_m = raw_depth.astype(np.float32) * self.depth_scale
        h, w = depth_m.shape

        color_frame = frames.get_color_frame() if not self.depth_only_mode else None
        if color_frame:
            color_bgr = np.asanyarray(color_frame.get_data()).copy()
            if (color_bgr.shape[1], color_bgr.shape[0]) != (w, h):
                color_bgr = cv2.resize(color_bgr, (w, h), interpolation=cv2.INTER_LINEAR)
        else:
            norm = np.clip((depth_m - 0.25) / 3.5, 0.0, 1.0)
            gray = ((1.0 - norm) * 220.0).astype(np.uint8)
            color_bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

        if (w, h) != (self.intrinsics.width, self.intrinsics.height):
            sx = w / float(self.intrinsics.width)
            sy = h / float(self.intrinsics.height)
            self.intrinsics = CameraIntrinsics(
                width=w,
                height=h,
                fx=self.intrinsics.fx * sx,
                fy=self.intrinsics.fy * sy,
                ppx=self.intrinsics.ppx * sx,
                ppy=self.intrinsics.ppy * sy,
                depth_scale=self.depth_scale,
            )

        return FrameBundle(
            timestamp=time.time(),
            color_bgr=color_bgr,
            depth_m=depth_m,
            intrinsics=self.intrinsics,
            imu_yaw=0.0,
            imu_pitch=self.config.nominal_camera_pitch_deg,
            imu_roll=0.0,
        )

    def read_frame(self, timeout_ms: int = 1500) -> FrameBundle:
        if self.pipeline is None or self.intrinsics is None:
            raise RuntimeError("La cámara RealSense D415 no ha sido iniciada con start().")

        try:
            frames = self.pipeline.wait_for_frames(timeout_ms=timeout_ms)
            if frames and frames.get_depth_frame():
                self._last_bundle = self._convert_frames(frames)
                return self._last_bundle
        except Exception:
            if self._last_bundle is not None:
                return FrameBundle(
                    timestamp=time.time(),
                    color_bgr=self._last_bundle.color_bgr.copy(),
                    depth_m=self._last_bundle.depth_m.copy(),
                    intrinsics=self._last_bundle.intrinsics,
                    imu_yaw=self._last_bundle.imu_yaw,
                    imu_pitch=self._last_bundle.imu_pitch,
                    imu_roll=self._last_bundle.imu_roll,
                )
            raise

        if self._last_bundle is not None:
            return self._last_bundle
        raise RuntimeError("Frame vacío desde la RealSense D415.")

    def stop(self) -> None:
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
