"""Adaptador de captura en tiempo real para Intel RealSense Depth Camera D415."""

from __future__ import annotations

import time
from typing import Optional

import numpy as np

from src.config import PipelineConfig
from src.models import CameraIntrinsics, FrameBundle

try:
    import pyrealsense2 as rs
except ImportError:  # pragma: no cover
    rs = None


class RealSenseD415Camera:
    """Gestiona el stream RGB-D alineado de la Intel RealSense D415 por USB-C."""

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
        self._spatial_filter = None
        self._temporal_filter = None
        self._hole_filling_filter = None

    @staticmethod
    def list_connected_devices() -> list[str]:
        """Devuelve los nombres de las cámaras RealSense conectadas al sistema."""
        if rs is None:
            return []
        ctx = rs.context()
        return [dev.get_info(rs.camera_info.name) for dev in ctx.query_devices()]

    def start(self) -> None:
        """Inicia el pipeline de `pyrealsense2` con fallback automático de perfil USB."""
        if rs is None:
            raise RuntimeError("El paquete 'pyrealsense2' no está instalado.")

        devices = self.list_connected_devices()
        if not devices:
            raise RuntimeError(
                "No se detectó ninguna cámara Intel RealSense conectada por USB."
            )
        self.device_name = devices[0]

        candidate_profiles = [
            (self.config.width, self.config.height, self.config.fps),
            (640, 480, 30),
            (640, 480, 15),
            (424, 240, 30),
            (424, 240, 15),
        ]

        last_err: Optional[Exception] = None
        profile = None

        for w, h, fps in candidate_profiles:
            try:
                pipeline = rs.pipeline()
                rs_cfg = rs.config()
                rs_cfg.enable_stream(rs.stream.depth, w, h, rs.format.z16, fps)
                rs_cfg.enable_stream(rs.stream.color, w, h, rs.format.bgr8, fps)
                profile = pipeline.start(rs_cfg)
                self.pipeline = pipeline
                break
            except Exception as exc:
                last_err = exc
                continue

        if profile is None or self.pipeline is None:
            raise RuntimeError(
                f"No fue posible iniciar el stream de {self.device_name}: {last_err}"
            )

        depth_sensor = profile.get_device().first_depth_sensor()
        self.depth_scale = float(depth_sensor.get_depth_scale())
        self.align = rs.align(rs.stream.color)

        color_stream = profile.get_stream(rs.stream.color).as_video_stream_profile()
        intr = color_stream.get_intrinsics()
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

        # Descartar primeros frames de autoexposición para estabilizar el IR
        for _ in range(5):
            try:
                self.pipeline.wait_for_frames(timeout_ms=2000)
            except Exception:
                break

    def read_frame(self, timeout_ms: int = 3000) -> FrameBundle:
        """Captura y devuelve un `FrameBundle` con color BGR y profundidad en metros."""
        if self.pipeline is None or self.align is None or self.intrinsics is None:
            raise RuntimeError("La cámara RealSense D415 no ha sido iniciada con start().")

        frames = self.pipeline.wait_for_frames(timeout_ms=timeout_ms)
        aligned_frames = self.align.process(frames)

        depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()
        if not depth_frame or not color_frame:
            raise RuntimeError("Frame incompleto recibido desde la RealSense D415.")

        if self.use_hardware_filters:
            if self._spatial_filter is not None:
                depth_frame = self._spatial_filter.process(depth_frame)
            if self._temporal_filter is not None:
                depth_frame = self._temporal_filter.process(depth_frame)
            if self._hole_filling_filter is not None:
                depth_frame = self._hole_filling_filter.process(depth_frame)

        raw_depth = np.asanyarray(depth_frame.get_data())
        color_bgr = np.asanyarray(color_frame.get_data()).copy()
        depth_m = raw_depth.astype(np.float32) * self.depth_scale

        # Si los filtros cambiaron resolución, actualizar intrínsecos
        h, w = depth_m.shape
        if (w, h) != (self.intrinsics.width, self.intrinsics.height):
            color_bgr = color_bgr[:h, :w]

        return FrameBundle(
            timestamp=time.time(),
            color_bgr=color_bgr,
            depth_m=depth_m,
            intrinsics=self.intrinsics,
            imu_yaw=0.0,
            imu_pitch=self.config.nominal_camera_pitch_deg,
            imu_roll=0.0,
        )

    def stop(self) -> None:
        """Detiene el pipeline de `pyrealsense2` limpiamente."""
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
