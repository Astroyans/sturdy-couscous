from __future__ import annotations

import heapq
import math
from typing import Optional

import numpy as np

from src.config import PipelineConfig
from src.models import (
    GRID_BOULDER,
    GRID_CREVICE,
    GRID_FREE,
    GRID_STEEP,
    GRID_UNKNOWN,
    NavigationResult,
)
from src.navigation.occupancy_grid import OccupancyGridBuilder
from src.odometry.visual_odometry import Pose2D


class PathPlanner:
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

    def __init__(
        self,
        config: Optional[PipelineConfig] = None,
        grid_builder: Optional[OccupancyGridBuilder] = None,
    ) -> None:
        self.config = config or PipelineConfig()
        self.grid_builder = grid_builder or OccupancyGridBuilder(self.config)

    @staticmethod
    def _is_blocked(cell_value: int) -> bool:
        return cell_value in (GRID_BOULDER, GRID_CREVICE, GRID_STEEP)

    def _select_frontier_goal_cell(
        self,
        grid: np.ndarray,
        preferred_col: int,
        min_row: int = 0,
    ) -> tuple[int, int]:
        rows, cols = grid.shape
        for r in range(rows - 2, max(1, min_row), -1):
            for offset in range(cols // 2):
                for c in (preferred_col - offset, preferred_col + offset):
                    if 0 <= c < cols and int(grid[r, c]) == GRID_FREE:
                        return (r, c)
            for offset in range(cols // 2):
                for c in (preferred_col - offset, preferred_col + offset):
                    if (
                        0 <= c < cols
                        and int(grid[r, c]) != GRID_UNKNOWN
                        and not self._is_blocked(int(grid[r, c]))
                    ):
                        return (r, c)
        return (min_row, preferred_col)

    def _run_search(
        self,
        occupancy_grid: np.ndarray,
        cost_grid: np.ndarray,
        start: tuple[int, int],
        goal: tuple[int, int],
        use_heuristic: bool,
    ) -> tuple[list[tuple[int, int]], int, tuple[int, int]]:
        rows, cols = occupancy_grid.shape
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
                    h = math.hypot(goal[0] - nr, goal[1] - nc) if use_heuristic else 0.0
                    heapq.heappush(open_heap, (tentative_g + h, nr, nc))

        path_cells: list[tuple[int, int]] = []
        curr = best_reached
        while curr in came_from:
            path_cells.append(curr)
            curr = came_from[curr]
        path_cells.append(start)
        path_cells.reverse()
        return path_cells, explored, best_reached

    def _compute_route_for_mode(
        self,
        occupancy_grid: np.ndarray,
        cost_grid: np.ndarray,
        base_cell: tuple[int, int],
        current_cell: tuple[int, int],
        frontier_goal: tuple[int, int],
        navigation_mode: str,
        use_heuristic: bool,
    ) -> tuple[list[tuple[int, int]], int, tuple[int, int]]:
        if navigation_mode == "RETURN_TO_BASE":
            start_search = current_cell if current_cell != base_cell else frontier_goal
            goal_search = base_cell
            return self._run_search(
                occupancy_grid, cost_grid, start_search, goal_search, use_heuristic
            )

        if current_cell != base_cell:
            path_base_to_curr, exp1, _ = self._run_search(
                occupancy_grid, cost_grid, base_cell, current_cell, use_heuristic
            )
            path_curr_to_goal, exp2, reached = self._run_search(
                occupancy_grid, cost_grid, current_cell, frontier_goal, use_heuristic
            )
            return path_base_to_curr[:-1] + path_curr_to_goal, exp1 + exp2, reached

        return self._run_search(
            occupancy_grid, cost_grid, base_cell, frontier_goal, use_heuristic
        )

    def plan(
        self,
        occupancy_grid: np.ndarray,
        cost_grid: np.ndarray,
        safe_corridor: bool,
        algorithm: Optional[str] = None,
        pose: Optional[Pose2D] = None,
        navigation_mode: str = "EXPLORE_FORWARD",
    ) -> NavigationResult:
        planner_name = (algorithm or self.config.default_planner).lower().strip()
        if planner_name not in ("astar", "dijkstra"):
            planner_name = "astar"

        rows, cols = occupancy_grid.shape
        base_cell = (0, cols // 2)
        current_pose = pose or Pose2D()
        current_cell = self.grid_builder.world_to_cell(current_pose.x_m, current_pose.y_m)
        frontier_goal = self._select_frontier_goal_cell(
            occupancy_grid, current_cell[1], min_row=current_cell[0]
        )

        # Ejecutar automaticamente ambos algoritmos (A* y Dijkstra) sobre el mapa acumulado
        astar_cells, astar_exp, astar_reached = self._compute_route_for_mode(
            occupancy_grid,
            cost_grid,
            base_cell,
            current_cell,
            frontier_goal,
            navigation_mode,
            use_heuristic=True,
        )
        dijkstra_cells, dijkstra_exp, dijkstra_reached = self._compute_route_for_mode(
            occupancy_grid,
            cost_grid,
            base_cell,
            current_cell,
            frontier_goal,
            navigation_mode,
            use_heuristic=False,
        )

        if planner_name == "astar":
            path_cells = astar_cells
            explored = astar_exp
            reached = astar_reached
        else:
            path_cells = dijkstra_cells
            explored = dijkstra_exp
            reached = dijkstra_reached

        path_world_xy = [
            self.grid_builder.cell_to_world(r, c) for (r, c) in path_cells
        ]

        recommended_heading_deg = 0.0
        if len(path_world_xy) >= 2:
            ahead_pts = [
                (wx, wy)
                for (wx, wy) in path_world_xy
                if math.hypot(wx - current_pose.x_m, wy - current_pose.y_m) >= 0.60
            ]
            target_wx, target_wy = ahead_pts[0] if ahead_pts else path_world_xy[-1]
            recommended_heading_deg = math.degrees(
                math.atan2(
                    target_wx - current_pose.x_m,
                    max(target_wy - current_pose.y_m, 1e-3),
                )
            )

        res = self.config.grid_resolution_m
        explored_cells = int(np.sum(occupancy_grid != GRID_UNKNOWN))
        explored_area_m2 = explored_cells * (res * res)

        return NavigationResult(
            occupancy_grid=occupancy_grid,
            cost_grid=cost_grid,
            path_cells=path_cells,
            path_world_xy=path_world_xy,
            astar_path_cells=astar_cells,
            dijkstra_path_cells=dijkstra_cells,
            astar_explored_nodes=astar_exp,
            dijkstra_explored_nodes=dijkstra_exp,
            safe_corridor=safe_corridor,
            recommended_heading_deg=recommended_heading_deg,
            planner_used=planner_name,
            explored_nodes=explored,
            start_cell=base_cell,
            current_cell=current_cell,
            goal_cell=reached,
            trajectory_cells=list(self.grid_builder.trajectory_cells),
            explored_area_m2=explored_area_m2,
            total_Persistent_obstacles=len(self.grid_builder.global_obstacles),
            navigation_mode=navigation_mode,
        )
