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

    def _build_nav_cost(
        self,
        occupancy_grid: np.ndarray,
        cost_grid: np.ndarray,
    ) -> list[float]:
        blocked = (
            (occupancy_grid == GRID_BOULDER)
            | (occupancy_grid == GRID_CREVICE)
            | (occupancy_grid == GRID_STEEP)
            | (~np.isfinite(cost_grid))
        )
        return np.where(blocked, -1.0, cost_grid).ravel().tolist()

    def _run_search(
        self,
        occupancy_grid: np.ndarray,
        cost_grid: np.ndarray,
        start: tuple[int, int],
        goal: tuple[int, int],
        use_heuristic: bool,
        nav_cost: Optional[list[float]] = None,
    ) -> tuple[list[tuple[int, int]], int, tuple[int, int]]:
        rows, cols = occupancy_grid.shape
        total_cells = rows * cols
        start_idx = start[0] * cols + start[1]
        goal_idx = goal[0] * cols + goal[1]
        goal_r, goal_c = goal

        if nav_cost is None:
            nav_cost = self._build_nav_cost(occupancy_grid, cost_grid)

        inf = float("inf")
        g_score = [inf] * total_cells
        g_score[start_idx] = 0.0
        came_from = [-1] * total_cells
        visited = [False] * total_cells

        open_heap: list[tuple[float, int]] = [(0.0, start_idx)]
        explored = 0
        best_reached_idx = start_idx
        best_dist_sq = (goal_r - start[0]) ** 2 + (goal_c - start[1]) ** 2
        hypot = math.hypot
        heappush = heapq.heappush
        heappop = heapq.heappop
        neighbors = self._NEIGHBORS

        while open_heap:
            _, idx = heappop(open_heap)
            if visited[idx]:
                continue
            visited[idx] = True
            explored += 1

            r, c = divmod(idx, cols)
            dr_g = goal_r - r
            dc_g = goal_c - c
            d_sq = dr_g * dr_g + dc_g * dc_g
            if d_sq < best_dist_sq:
                best_dist_sq = d_sq
                best_reached_idx = idx

            if idx == goal_idx:
                best_reached_idx = goal_idx
                break

            base_g = g_score[idx]
            for dr, dc, step_len in neighbors:
                nr = r + dr
                nc = c + dc
                if 0 <= nr < rows and 0 <= nc < cols:
                    nidx = nr * cols + nc
                    if visited[nidx]:
                        continue
                    cell_cost = nav_cost[nidx]
                    if cell_cost < 0.0:
                        continue
                    tentative_g = base_g + step_len * cell_cost
                    if tentative_g < g_score[nidx]:
                        g_score[nidx] = tentative_g
                        came_from[nidx] = idx
                        h = hypot(goal_r - nr, goal_c - nc) if use_heuristic else 0.0
                        heappush(open_heap, (tentative_g + h, nidx))

        path_cells: list[tuple[int, int]] = []
        curr = best_reached_idx
        while curr != -1 and curr != start_idx:
            path_cells.append(divmod(curr, cols))
            curr = came_from[curr]
        path_cells.append(start)
        path_cells.reverse()
        return path_cells, explored, divmod(best_reached_idx, cols)

    def _compute_route_for_mode(
        self,
        occupancy_grid: np.ndarray,
        cost_grid: np.ndarray,
        base_cell: tuple[int, int],
        current_cell: tuple[int, int],
        frontier_goal: tuple[int, int],
        navigation_mode: str,
        use_heuristic: bool,
        nav_cost: Optional[list[float]] = None,
    ) -> tuple[list[tuple[int, int]], int, tuple[int, int]]:
        if navigation_mode == "RETURN_TO_BASE":
            start_search = current_cell if current_cell != base_cell else frontier_goal
            goal_search = base_cell
            return self._run_search(
                occupancy_grid, cost_grid, start_search, goal_search, use_heuristic, nav_cost=nav_cost
            )

        if current_cell != base_cell:
            path_base_to_curr, exp1, _ = self._run_search(
                occupancy_grid, cost_grid, base_cell, current_cell, use_heuristic, nav_cost=nav_cost
            )
            path_curr_to_goal, exp2, reached = self._run_search(
                occupancy_grid, cost_grid, current_cell, frontier_goal, use_heuristic, nav_cost=nav_cost
            )
            return path_base_to_curr[:-1] + path_curr_to_goal, exp1 + exp2, reached

        return self._run_search(
            occupancy_grid, cost_grid, base_cell, frontier_goal, use_heuristic, nav_cost=nav_cost
        )

    def plan(
        self,
        occupancy_grid: np.ndarray,
        cost_grid: np.ndarray,
        safe_corridor: bool,
        algorithm: Optional[str] = None,
        pose: Optional[Pose2D] = None,
        navigation_mode: str = "EXPLORE_FORWARD",
        goal_world_xy: Optional[tuple[float, float]] = None,
    ) -> NavigationResult:
        planner_name = (algorithm or self.config.default_planner).lower().strip()
        if planner_name not in ("astar", "dijkstra"):
            planner_name = "astar"

        rows, cols = occupancy_grid.shape
        base_cell = (0, cols // 2)
        current_pose = pose or Pose2D()
        current_cell = self.grid_builder.world_to_cell(current_pose.x_m, current_pose.y_m)

        if goal_world_xy is not None:
            target_cell = self.grid_builder.world_to_cell(goal_world_xy[0], goal_world_xy[1])
            nav_cost = self._build_nav_cost(occupancy_grid, cost_grid)
            astar_cells, astar_exp, astar_reached = self._run_search(
                occupancy_grid,
                cost_grid,
                current_cell,
                target_cell,
                use_heuristic=True,
                nav_cost=nav_cost,
            )
            dijkstra_cells, dijkstra_exp, dijkstra_reached = self._run_search(
                occupancy_grid,
                cost_grid,
                current_cell,
                target_cell,
                use_heuristic=False,
                nav_cost=nav_cost,
            )
        elif navigation_mode == "RETURN_TO_BASE" or self.config.auto_frontier_route:
            frontier_goal = self._select_frontier_goal_cell(
                occupancy_grid, current_cell[1], min_row=current_cell[0]
            )
            nav_cost = self._build_nav_cost(occupancy_grid, cost_grid)
            astar_cells, astar_exp, astar_reached = self._compute_route_for_mode(
                occupancy_grid,
                cost_grid,
                base_cell,
                current_cell,
                frontier_goal,
                navigation_mode,
                use_heuristic=True,
                nav_cost=nav_cost,
            )
            dijkstra_cells, dijkstra_exp, dijkstra_reached = self._compute_route_for_mode(
                occupancy_grid,
                cost_grid,
                base_cell,
                current_cell,
                frontier_goal,
                navigation_mode,
                use_heuristic=False,
                nav_cost=nav_cost,
            )
        else:
            astar_cells, astar_exp, astar_reached = [], 0, current_cell
            dijkstra_cells, dijkstra_exp, dijkstra_reached = [], 0, current_cell

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
        astar_world_xy = [
            self.grid_builder.cell_to_world(r, c) for (r, c) in astar_cells
        ]
        dijkstra_world_xy = [
            self.grid_builder.cell_to_world(r, c) for (r, c) in dijkstra_cells
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
            astar_path_world_xy=astar_world_xy,
            dijkstra_path_world_xy=dijkstra_world_xy,
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
