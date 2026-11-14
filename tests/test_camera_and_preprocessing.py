"""Pruebas unitarias para Task 2: cámara sintética, preprocesamiento y detección D415."""

import numpy as np

from src.camera.realsense_d415 import RealSenseD415Camera
from src.camera.synthetic import SyntheticTerrainCamera
from src.config import PipelineConfig
from src.preprocessing.filters import clean_binary_mask, preprocess_depth


def test_synthetic_camera_generates_valid_rgbd_bundle():
    cfg = PipelineConfig(width=320, height=240)
    cam = SyntheticTerrainCamera(cfg, terrain_incline_deg=5.0, add_noise=True)
    frame = cam.read_frame()

    assert frame.color_bgr.shape == (240, 320, 3)
    assert frame.color_bgr.dtype == np.uint8
    assert frame.depth_m.shape == (240, 320)
    assert frame.depth_m.dtype == np.float32
    assert float(np.min(frame.depth_m)) >= cfg.min_depth_m
    assert float(np.max(frame.depth_m)) <= cfg.max_depth_m


def test_preprocess_depth_fills_small_holes_and_reduces_noise():
    cfg = PipelineConfig(width=160, height=120)
    raw = np.full((120, 160), 1.50, dtype=np.float32)
    # Inyectar un pequeño hueco inválido (0.0) en el centro
    raw[60, 80] = 0.0
    raw[60, 81] = 0.0

    filtered, valid_mask = preprocess_depth(raw, cfg)
    assert valid_mask[60, 80]
    assert np.isclose(filtered[60, 80], 1.50, atol=0.02)


def test_clean_binary_mask_removes_isolated_speckles():
    noisy = np.zeros((100, 100), dtype=bool)
    # Objeto real grande
    noisy[30:60, 30:60] = True
    # Píxel aislado de ruido
    noisy[10, 10] = True

    cleaned = clean_binary_mask(noisy, kernel_size=5, open_iterations=1, close_iterations=1)
    assert not cleaned[10, 10]
    assert cleaned[45, 45]


def test_realsense_device_enumeration_works():
    devices = RealSenseD415Camera.list_connected_devices()
    assert isinstance(devices, list)
