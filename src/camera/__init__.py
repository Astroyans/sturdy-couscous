"""Fuentes de captura RGB-D (Intel RealSense D415 USB-C y Simulador Sintético 3D)."""

from src.camera.realsense_d415 import RealSenseD415Camera
from src.camera.synthetic import SyntheticTerrainCamera

__all__ = ["RealSenseD415Camera", "SyntheticTerrainCamera"]
