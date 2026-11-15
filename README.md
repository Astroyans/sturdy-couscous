# sturdy-couscous — Motor de Análisis Topográfico Aéreo y Navegación Autónoma (Intel RealSense D415)

Repositorio oficial del módulo de visión por computadora, mapeo topográfico persistente y planificación de rutas (**A\*** y **Dijkstra**) para el **Dron / Rover** (*NASA Space Apps Challenge 2026 — Interplanetary Survival Guide: Martian Map*).

## Flujo General del Sistema (3 Etapas)

1. **Dron (Intel RealSense D415 a `-90°` cenital + Raspberry Pi 3 / USB 3.2):** captura en vuelo (5–6 m) los cuadros crudos RGB y de profundidad.
2. **Laptop (`main_expo.py` / `main.py`):** ejecuta la alineación píxel-a-píxel RGB-D (`rs.align`), reconstrucción métrica 3D, ajuste del plano del suelo (**RANSAC + SVD** con suavizado EMA), segmentación de obstáculos 3D (`BOULDER` y `CREVICE` con **DBSCAN**), odometría visual 3D, mapa global persistente (**Log-Odds**) y cálculo de rutas **A\*** + **Dijkstra** bajo demanda cuando el astronauta selecciona un destino.
3. **iPad A16 (Expo Go Dashboard):** se conecta directo por Wi-Fi LAN a la laptop (`ExpoLANServer`) o vía backend externo para visualizar en tiempo real el JSON de telemetría y las **3 imágenes base64** (`rgb`, `depth`, `route_map`), enviando las coordenadas del tap en el mapa para trazar la ruta desde la posición actual del astronauta.

## Puntos de Entrada (`main_expo.py` y `main.py`)

### 1. Nodo Real para Expo Go en iPad A16 (`main_expo.py` — sin ventana OpenCV)
```powershell
# Levanta servidor LAN unificado (HTTP + WebSocket en :8765) recibiendo desde Raspi 3 (UDP :8766)
python main_expo.py --source raspi3 --raspi-proto udp --raspi-port 8766

# Enviar además a un backend externo (HTTP POST o WebSocket)
python main_expo.py --transport http --url http://<IP-BACKEND>:8764/api/ingest
python main_expo.py --transport ws --url ws://<IP-BACKEND>:8764
```
Puertos y endpoints expuestos automáticamente en la IP LAN de la laptop:
- `ws://<IP-LAPTOP>:8765` — Streaming WebSocket en vivo (`node`, `timestamp`, `data`, `images.rgb`, `images.depth`, `images.route_map`) y recepción de destino.
- `GET http://<IP-LAPTOP>:8765/latest_expo_payload.json` — Último paquete completo JSON + 3 imágenes JPEG en `data:image/jpeg;base64,...`.
- `GET http://<IP-LAPTOP>:8765/map_state.json` — Estado del mapa topográfico persistente.
- `POST http://<IP-LAPTOP>:8765/api/goal` — Fijar destino al tocar el mapa en el iPad (`{"norm_x": 0.5, "norm_y": 0.2}`, `{"x_m": 0.5, "y_m": 1.2}` o `{"clear": true}`).
- **Puertos dedicados sin conflicto:** `8764` (ingesta Dashboard `/api/ingest`), `8765` (ExpoLANServer HTTP+WS), `8766` (entrada Raspi 3 UDP/TCP), `8767` (UDP RPi Centro de Mando), `8769` (UDP Dispositivo Astronauta).

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
