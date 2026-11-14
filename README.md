# sturdy-couscous — Motor de Análisis Topográfico y Navegación Autónoma (Intel RealSense D415)

Repositorio oficial del módulo de visión por computadora y navegación táctica del **Rover** (*NASA Space Apps Challenge 2026 — Interplanetary Survival Guide: Martian Map*).

## Arquitectura del Pipeline (5 Fases)

1. **Fase 1 — Adquisición D415 y Preprocesamiento (`src/camera/`, `src/preprocessing/`):**
   - Captura RGB-D alineada desde la **Intel RealSense Depth Camera D415** por USB-C (con negociación automática de ancho de banda USB 3.x / USB 2.1) o desde el simulador matemático 3D (`SyntheticTerrainCamera`).
   - Enmascaramiento de rango físico (`0.25 m` – `4.00 m`), cierre morfológico (dilatación + erosión) para rellenar sombras IR y suavizado Gaussiano normalizado.
2. **Fase 2 — Reconstrucción 3D, Plano del Suelo (RANSAC) e Inclinación (`src/terrain/`):**
   - Deproyección métrica 3D $(X_{\text{right}}, Y_{\text{forward}}, Z_{\text{up}})$ compensando el ángulo de cabeceo de la cámara (`-15°`).
   - Ajuste dinámico del plano dominante del suelo mediante **RANSAC vectorizado** + refinamiento SVD, obteniendo `altitude_m`, `max_incline_deg`, mapa de pendientes locales y elevación firmada $\Delta h$.
3. **Fase 3 — Detección y Clasificación de Obstáculos `BOULDER` y `CREVICE` (`src/obstacles/`):**
   - Segmentación por umbrales sobre $\Delta h$, limpieza morfológica (apertura y cierre) y agrupamiento espacial 3D con **DBSCAN** (`scikit-learn`) + componentes conexos.
   - Extracción de `distance_m`, `bearing_deg`, `height_m` (`BOULDER`) y `depth_m` (`CREVICE`).
4. **Fase 4 — Malla de Ocupación 2D (*Bird's-Eye View*) y Ruta Mínima A\* / Dijkstra (`src/navigation/`):**
   - Proyección cenital 2D, inflación por el radio de seguridad del Rover, mapa de costos por transformada de distancia, evaluación de `safe_corridor` y búsqueda de camino óptimo mediante **A\*** y **Dijkstra**.
5. **Fase 5 — HUD Táctico en Tiempo Real y Contrato JSON `drone_cv` (`src/visualization/`, `main.py`):**
   - Interfaz en vivo `1280x720` con 4 paneles: Vista RGB con ruta proyectada, Mapa de Calor de Profundidad y Segmentación, Radar 2D de Malla de Ocupación con traza de ruta, y Barra de Telemetría `drone_cv`.

## Ejecución Rápida

```powershell
# Ejecutar en vivo con la Intel RealSense D415 conectada por USB-C (abre ventana HUD)
python main.py

# Ejecutar con el simulador 3D de terreno marciano
python main.py --source synthetic

# Ajustar sensibilidad para maqueta sobre mesa (ej. rocas/grietas desde 3 cm)
python main.py --boulder-min-height 0.03 --crevice-min-depth 0.03

# Correr suite de pruebas automatizadas
python -m pytest -v
```

## Controles en Vivo dentro del HUD
- **`[P]`**: Alternar en vivo entre **A\*** y **Dijkstra**.
- **`[C]`**: Fijar / liberar el plano de referencia de la mesa/terreno.
- **`[S]`**: Guardar captura PNG del HUD y archivo JSON con el payload `drone_cv` en `captures/`.
- **`[Q]` / `[ESC]`**: Cerrar la aplicación.
