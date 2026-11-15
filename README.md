# sturdy-couscous — Aerial Topographic Vision & Autonomous Mars Navigation Engine

**NASA Space Apps Challenge 2026 — *Interplanetary Survival Guide: Martian Map*** (Team **Astroyans**).

Computer vision, persistent 2D/3D topographic occupancy mapping, and autonomous path-planning engine (**A\*** and **Dijkstra**) powered by **Intel RealSense D415** RGB-D depth sensing for lunar/Martian EVA reconnaissance.

---

## System Pipeline (3 Stages)

1. **Recon Drone (`Intel RealSense D415` at `-90°` nadir + `Raspberry Pi 3`):** Captures aligned RGB and depth frames in flight and streams chunked UDP/TCP packets (`R3D1` / `R3U1`) to the ground compute node.
2. **Ground Compute Engine (`main_expo.py` / `main.py`):**
   * Pixel-aligned RGB-D preprocessing and depth filtering (`src/preprocessing/filters.py`).
   * Ground-plane estimation via **RANSAC + SVD** with Exponential Moving Average smoothing (`src/terrain/ground_plane.py`).
   * 3D obstacle segmentation (`BOULDER` and `CREVICE`) using **DBSCAN** clustering (`src/obstacles/detector.py`).
   * 3D visual odometry (`src/odometry/visual_odometry.py`) and **Log-Odds** persistent global occupancy grid (`src/navigation/occupancy_grid.py`).
   * Cached on-demand **A\*** and **Dijkstra** path planning (`src/navigation/path_planner.py`), trajectory prediction & unexplored-quadrant patch injection (`src/navigation/trajectory_predictor.py`).
3. **Tactical Dashboards & Mesh Backup:**
   * Serves real-time JSON telemetry and three JPEG Base64 visual feeds (`rgb`, `depth`, `route_map`) over WebSocket/HTTP (`:8765`) to the Web Mission Control and iPad Expo Go app ([Astroyans/full-stack](https://github.com/Astroyans/full-stack)).
   * Forwards compact terrain state and ESP32-compatible `/ruta.txt` waypoint strings over UDP (`:8767`) to the Command Center link (`src/transport/centro_mando_link.py`).

---

## Entry Points (`main_expo.py` & `main.py`)

### 1. Headless Unified Server for Web Dashboard & iPad Expo Go (`main_expo.py`)

```powershell
# Start unified LAN server (HTTP + WebSocket on :8765) receiving Raspi 3 chunks on UDP :8766
python main_expo.py --source raspi3 --raspi-proto udp --raspi-port 8766

# Optional: forward telemetry to an external ingest backend (HTTP POST or WebSocket)
python main_expo.py --transport http --url http://<BACKEND-IP>:8764/api/ingest
python main_expo.py --transport ws --url ws://<BACKEND-IP>:8764
```

### Exposed Endpoints & Ports
* `ws://<LAPTOP-IP>:8765` — Live WebSocket stream (`node`, `timestamp`, `data`, `images.rgb`, `images.depth`, `images.route_map`) and interactive `request_route` / `dispatch_drone` handling.
* `GET http://<LAPTOP-IP>:8765/latest_expo_payload.json` — Latest full telemetry snapshot + Base64 JPEG feeds (`data:image/jpeg;base64,...`).
* `GET http://<LAPTOP-IP>:8765/map_state.json` — Persistent global occupancy grid state.
* `POST http://<LAPTOP-IP>:8765/api/goal` — Set a navigation destination from the Web/iPad map (`{"norm_x": 0.5, "norm_y": 0.2}`, `{"x_m": 0.5, "y_m": 1.2}`, or `{"clear": true}`).
* **Dedicated Port Layout:** `8764` (external dashboard `/api/ingest`), `8765` (`ExpoLANServer` HTTP + WebSocket), `8766` (Raspberry Pi 3 UDP/TCP video+depth ingest), `8767` (Command Center UDP bridge), `8769` (Astronaut wearable/device UDP).

### 2. Support Scripts (`scripts/`)
* `scripts/centro_mando_server.py` — Command Center UDP listener (`:8767`) logging quadrant scans, predicted trajectories, and `/ruta.txt` ESP32 strings.
* `scripts/replay_mars_drone.py` — Replays recorded RealSense D415 captures with Mars dune relief synthesis for repeatable live demonstrations.
* `scripts/raspi3_sender.py` — Raspberry Pi 3 RGB-D UDP/TCP packet transmitter (`R3U1` / `R3D1`).

### 3. Local OpenCV Diagnostic HUD (`main.py`)

```powershell
# Run local 1280x720 OpenCV HUD with Intel RealSense D415 (nadir -90° view by default)
python main.py

# Run with built-in 3D synthetic Martian terrain simulator
python main.py --source synthetic
```

**HUD Controls:**
* **Left-click on 2D map:** Set target destination and compute **A\*** and **Dijkstra** paths from current position.
* **Right-click / `[X]`:** Clear active destination.
* **`[P]`:** Toggle primary planner (`A*` <-> `Dijkstra`).
* **`[M]`:** Toggle mode (`EXPLORE_FORWARD` <-> `RETURN_TO_BASE`).
* **`[C]`:** Lock / unlock ground plane reference.
* **`[R]`:** Reset persistent occupancy map.
* **`[S]`:** Save PNG + JSON snapshot to `captures/`.
* **`[Q]` / `[ESC]`:** Quit.

---

## Automated Tests

```powershell
pytest -v
```

---

## Related Astroyans Repositories

* **Web Dashboard & Astronaut iPad App:** [Astroyans/full-stack](https://github.com/Astroyans/full-stack)
* **Rust Telemetry & ToF Streaming Link (`nodo_dron` / `estacion_tierra`):** [Astroyans/Dron](https://github.com/Astroyans/Dron)
* **ESP32 Mesh & MicroSD Route Backup Firmware:** [Astroyans/Config-ESP32](https://github.com/Astroyans/Config-ESP32)
