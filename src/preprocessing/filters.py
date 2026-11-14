"""Filtros Gaussianos normalizados y operaciones morfológicas (erosión y dilatación)."""

from __future__ import annotations

import cv2
import numpy as np

from src.config import PipelineConfig


def preprocess_depth(
    depth_m: np.ndarray,
    config: PipelineConfig,
) -> tuple[np.ndarray, np.ndarray]:
    """Filtra el mapa de profundidad aplicando cierre morfológico y suavizado Gaussiano normalizado.

    Implementa la primera etapa metodológica de la tesis:
    1. Enmascaramiento de lecturas fuera del rango físico de la D415 (`min_depth_m` a `max_depth_m`).
    2. Cierre morfológico (dilatación seguida de erosión) para rellenar micro-huecos de sombra IR.
    3. Suavizado Gaussiano normalizado sobre píxeles válidos para atenuar el ruido de cuantización
       sin contaminar bordes con ceros de píxeles inválidos.

    Returns:
        filtered_depth_m: Mapa de profundidad filtrado (float32, en metros; 0.0 donde es inválido).
        valid_mask: Máscara booleana (H, W) de píxeles con lectura válida.
    """
    valid_mask = (
        np.isfinite(depth_m)
        & (depth_m >= config.min_depth_m)
        & (depth_m <= config.max_depth_m)
    )

    clean_depth = np.where(valid_mask, depth_m, 0.0).astype(np.float32)

    # 1. Cierre morfológico para rellenar pequeños agujeros sin lectura (0.0)
    k_morph = max(3, int(config.morph_kernel_size) | 1)
    morph_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_morph, k_morph))

    if config.morph_close_iterations > 0:
        dilated = cv2.dilate(
            clean_depth, morph_kernel, iterations=config.morph_close_iterations
        )
        closed_depth = cv2.erode(
            dilated, morph_kernel, iterations=config.morph_close_iterations
        )
        # Rellenar únicamente huecos pequeños rodeados de lecturas válidas
        valid_u8 = valid_mask.astype(np.uint8) * 255
        closed_mask_u8 = cv2.morphologyEx(
            valid_u8,
            cv2.MORPH_CLOSE,
            morph_kernel,
            iterations=config.morph_close_iterations,
        )
        hole_fill_mask = (closed_mask_u8 > 0) & (~valid_mask) & (closed_depth >= config.min_depth_m)
        clean_depth = np.where(hole_fill_mask, closed_depth, clean_depth)
        valid_mask = valid_mask | hole_fill_mask

    # 2. Filtro Gaussiano normalizado para no promediar con ceros en bordes
    k_gauss = max(3, int(config.gaussian_kernel_size) | 1)
    weights = valid_mask.astype(np.float32)

    blurred_depth = cv2.GaussianBlur(
        clean_depth * weights,
        (k_gauss, k_gauss),
        config.gaussian_sigma,
    )
    blurred_weights = cv2.GaussianBlur(
        weights,
        (k_gauss, k_gauss),
        config.gaussian_sigma,
    )

    filtered_depth_m = np.where(
        valid_mask & (blurred_weights > 1e-4),
        blurred_depth / np.maximum(blurred_weights, 1e-4),
        0.0,
    ).astype(np.float32)

    return filtered_depth_m, valid_mask


def clean_binary_mask(
    mask: np.ndarray,
    kernel_size: int = 5,
    open_iterations: int = 1,
    close_iterations: int = 2,
) -> np.ndarray:
    """Aplica apertura (erosión -> dilatación) y cierre (dilatación -> erosión) a una máscara binaria."""
    k = max(3, int(kernel_size) | 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    mask_u8 = (mask.astype(np.uint8)) * 255

    if open_iterations > 0:
        eroded = cv2.erode(mask_u8, kernel, iterations=open_iterations)
        mask_u8 = cv2.dilate(eroded, kernel, iterations=open_iterations)

    if close_iterations > 0:
        dilated = cv2.dilate(mask_u8, kernel, iterations=close_iterations)
        mask_u8 = cv2.erode(dilated, kernel, iterations=close_iterations)

    return mask_u8 > 0
