"""Planificador de ruta mínima sobre Malla de Ocupación mediante A* y Dijkstra."""

from __future__ import annotations

import heapq
import math
from typing import Optional

import numpy as np

from src.config import PipelineConfig
from src.models import (
    GRID_BOULDER,
    GRID_CREVICE,
    GRID_STEEP,
    NavigationResult,
)
from src.navigation.occupancy_grid import OccupancyGridBuilder


class PathPlanner:
    """Implementa búsqueda de camino óptimo libre de colisiones con A* y Dijkstra."""

    # Vecindad 8-conexa: (d_row, d_col, step_distance)
    _NEIGHBORS: tuple[tuple[int, int, float], ...] = (
        (1, 0, 1.0),
        (0, -1, 1.0),
        (0, 1, 1.0),
        (-1, 0, 1.0),
        (1, -1, math.sqrt(2.0)),
        (1, 1, math.sqrt(2.0)),
        (-1, -1, math.sqrt(2.0)),
        (-1, 1, math.sqrt(2.0)),
    )

    def __init__(self, config: Optional[PipelineConfig] = None) -> None:
        self.config = config or PipelineConfig()
        self.grid_builder = OccupancyGridBuilder(self.config)

    @staticmethod
    def _is_blocked(cell_value: int) -> bool:
        return cell_value in (GRID_BOULDER, GRID_CREVICE, GRID_STEEP)

    def _select_goal_cell(self, grid: np.ndarray, start_col: int) -> tuple[int, int]:
        """Selecciona la celda meta más lejana y centrada que esté libre de obstáculos."""
        rows, cols = grid.shape
        for r in range(rows - 2, 1, -1):
            # Buscar desde el centro hacia los lados
            for offset in range(cols // 2):
                for c in (start_col - offset, start_col + offset):
                    if 0 <= c < cols and not self._is_blocked(int(grid[r, c])):
                        # Preferir celdas no infladas en la primera pasada
                        if int(grid[r, c]) == 0:
                            return (r, c)
            # Segunda pasada permitiendo celdas infladas si todo el frente está estrecho
            for offset in range(cols // 2):
                for c in (start_col - offset, start_col + offset):
                    if 0 <= c < cols and not self._is_blocked(int(grid[r, c])):
                        return (r, c)
        return (0, start_col)

    def plan(
        self,
        occupancy_grid: np.ndarray,
        cost_grid: np.ndarray,
        safe_corridor: bool,
        algorithm: Optional[str] = None,
    ) -> NavigationResult:
        """Calcula la ruta óptima usando `'astar'` o `'dijkstra'`."""
        planner_name = (algorithm or self.config.default_planner).lower().strip()
        if planner_name not in ("astar", "dijkstra"):
            planner_name = "astar"

        rows, cols = occupancy_grid.shape
        start = (0, cols // 2)
        goal = self._select_goal_cell(occupancy_grid, start[1])

        use_heuristic = planner_name == "astar"
        g_score = np.full((rows, cols), np.inf, dtype=np.float64)
        g_score[start] = 0.0

        came_from: dict[tuple[int, int], tuple[int, int]] = {}
        open_heap: list[tuple[float, int, int]] = [(0.0, start[0], start[1])]
        visited = np.zeros((rows, cols), dtype=bool)

        explored = 0
        best_reached = start
        best_dist_to_goal = math.hypot(goal[0] - start[0], goal[1] - start[1])

        while open_heap:
            _, r, c = heapq.heappop(open_heap)
            if visited[r, c]:
                continue
            visited[r, c] = True
            explored += 1

            d_goal = math.hypot(goal[0] - r, goal[1] - c)
            if d_goal < best_dist_to_goal:
                best_dist_to_goal = d_goal
                best_reached = (r, c)

            if (r, c) == goal:
                best_reached = goal
                break

            base_g = g_score[r, c]
            for dr, dc, step_len in self._NEIGHBORS:
                nr, nc = r + dr, c + dc
                if not (0 <= nr < rows and 0 <= nc < cols):
                    continue
                if visited[nr, nc]:
                    continue
                if self._is_blocked(int(occupancy_grid[nr, nc])):
                    continue

                cell_cost = float(cost_grid[nr, nc])
                if not math.isfinite(cell_cost):
                    continue

                tentative_g = base_g + step_len * cell_cost
                if tentative_g < g_score[nr, nc]:
                    g_score[nr, nc] = tentative_g
                    came_from[(nr, nc)] = (r, c)
                    if use_heuristic:
                        h = math.hypot(goal[0] - nr, goal[1] - nc)
                    else:
                        h = 0.0
                    heapq.heappush(open_heap, (tentative_g + h, nr, nc))

        # Reconstrucción del camino desde `best_reached` hasta `start`
        path_cells: list[tuple[int, int]] = []
        curr = best_reached
        while curr in came_from:
            path_cells.append(curr)
            curr = came_from[curr]
        path_cells.append(start)
        path_cells.reverse()

        path_world_xy = [
            self.grid_builder.cell_to_world(r, c) for (r, c) in path_cells
        ]

        # Calcular rumbo recomendado hacia un punto de anticipación (lookahead ~1.0 m)
        recommended_heading_deg = 0.0
        if len(path_world_xy) >= 2:
            lookahead_pt = path_world_xy[min(len(path_world_xy) - 1, 18)]
            recommended_heading_deg = math.degrees(
                math.atan2(lookahead_pt[0], max(lookahead_pt[1], 1e-3))
            )

        return NavigationResult(
            occupancy_grid=occupancy_grid,
            cost_grid=cost_grid,
            path_cells=path_cells,
            path_world_xy=path_world_xy,
            safe_corridor=safe_corridor,
            recommended_heading_deg=recommended_heading_deg,
            planner_used=planner_name,
            explored_nodes=explored,
            start_cell=start,
            goal_cell=best_reached,
        )
