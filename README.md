# sturdy-couscous — Motor de Análisis Topográfico Aéreo y Navegación Autónoma (Intel RealSense D415)

Repositorio oficial del módulo de visión por computadora, mapeo topográfico persistente y planificación de rutas (**A\*** y **Dijkstra**) para el **Dron / Rover** (*NASA Space Apps Challenge 2026 — Interplanetary Survival Guide: Martian Map*).

## Flujo General del Sistema (3 Etapas)

1. **Dron (Intel RealSense D415 a `-90°` cenital + Raspberry Pi 3 / USB 3.2):** captura en vuelo (5–6 m) los cuadros crudos RGB y de profundidad.
2. **Laptop (`main_expo.py` / `main.py`):** ejecuta la alineación píxel-a-píxel RGB-D (`rs.align`), reconstrucción métrica 3D, ajuste del plano del suelo (**RANSAC + SVD** con suavizado EMA), segmentación de obstáculos 3D (`BOULDER` y `CREVICE` con **DBSCAN**), odometría visual 3D, mapa global persistente (**Log-Odds**) y cálculo de rutas **A\*** + **Dijkstra** bajo demanda cuando el astronauta selecciona un destino.
3. **iPad A16 (Expo Go Dashboard):** se conecta directo por Wi-Fi LAN a la laptop (`ExpoLANServer`) o vía backend externo para visualizar en tiempo real el JSON de telemetría y las **3 imágenes base64** (`rgb`, `depth`, `route_map`), enviando las coordenadas del tap en el mapa para trazar la ruta desde la posición actual del astronauta.

## Puntos de Entrada (`main_expo.py` y `main.py`)

### 1. Nodo Real para Expo Go en iPad A16 (`main_expo.py` — sin ventana OpenCV)
```powershell
# Levanta servidor LAN directo hacia el iPad (HTTP :8000 y WebSocket :8001) con cámara a -90°
python main_expo.py

# Enviar además a un backend externo (HTTP POST o WebSocket)
python main_expo.py --transport http --url http://<IP-BACKEND>:3000/api/telemetry
python main_expo.py --transport ws --url ws://<IP-BACKEND>:3000/ws
```
Endpoints locales expuestos automáticamente en la IP LAN de la laptop:
- `ws://<IP-LAPTOP>:8001` — Streaming WebSocket en vivo (`node`, `timestamp`, `data`, `images.rgb`, `images.depth`, `images.route_map`) y recepción de destino.
- `GET http://<IP-LAPTOP>:8000/api/latest` — Último paquete completo JSON + 3 imágenes JPEG en `data:image/jpeg;base64,...`.
- `POST http://<IP-LAPTOP>:8000/api/goal` — Fijar destino al tocar el mapa en el iPad (`{"norm_x": 0.5, "norm_y": 0.2}`, `{"x_m": 0.5, "y_m": 1.2}` o `{"clear": true}`).

### 2. Dashboard Local de Diagnóstico OpenCV (`main.py`)
```powershell
# Ejecutar HUD local 1280x720 con la Intel RealSense D415 (vista aérea -90° por defecto)
python main.py

# Ejecutar con el simulador 3D de terreno marciano
python main.py --source synthetic
```
Controles en el HUD:
- **Click izquierdo en el mapa 2D**: fijar destino y calcular rutas **A\*** y **Dijkstra** desde la posición actual.
- **Click derecho / `[X]`**: limpiar destino activo.
- **`[P]`**: alternar algoritmo primario (`A*` ↔ `Dijkstra`).
- **`[M]`**: alternar modo (`EXPLORE_FORWARD` ↔ `RETURN_TO_BASE`).
- **`[C]`**: fijar / liberar plano del suelo.
- **`[R]`**: reiniciar mapa persistente.
- **`[S]`**: guardar snapshot PNG + JSON en `captures/`.
- **`[Q]` / `[ESC]`**: salir.

## Pruebas Automatizadas
```powershell
pytest -v
```
