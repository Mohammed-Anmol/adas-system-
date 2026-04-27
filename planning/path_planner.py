"""
Path planner using A* on an occupancy grid derived from the drivable mask
and obstacle detections. Adapted from PythonRobotics.
"""
import numpy as np
import heapq
import cv2


class PathPlanner:
    """A* grid planner operating on a bird's-eye-view occupancy grid."""

    GRID_RES = 10  # pixels per grid cell

    def __init__(self, config=None):
        self.grid_res = self.GRID_RES
        print("[Planner] A* Path Planner initialized.")

    def plan_path(self, drivable_mask, obstacles, ego_state):
        """Plan a path from bottom-center to a goal ahead.

        Parameters
        ----------
        drivable_mask : np.ndarray uint8 (H, W) — 255 = drivable
        obstacles     : list of dicts {x1, y1, x2, y2, ...}
        ego_state     : dict with ego_speed, ego_yaw, etc.

        Returns
        -------
        list of (x, y) pixel waypoints from start to goal
        """
        if drivable_mask is None:
            return []

        h, w = drivable_mask.shape[:2]
        grid = self._build_grid(drivable_mask, obstacles, h, w)

        # Start = bottom center, Goal = top center
        gr, gc = grid.shape
        start = (gr - 1, gc // 2)
        goal = (gr // 4, gc // 2)  # aim for upper quarter

        # Clamp goal to free cell
        goal = self._nearest_free(grid, goal)
        if goal is None or grid[start[0], start[1]] == 1:
            return []

        path_grid = self._astar(grid, start, goal)

        # Convert grid coords back to pixel coords
        path_px = [(c * self.grid_res + self.grid_res // 2,
                     r * self.grid_res + self.grid_res // 2)
                    for r, c in path_grid]
        return path_px

    def _build_grid(self, drivable_mask, obstacles, h, w):
        """Build binary occupancy grid: 0 = free, 1 = blocked."""
        gr = h // self.grid_res
        gc = w // self.grid_res
        grid = np.ones((gr, gc), dtype=np.uint8)  # default blocked

        # Mark drivable cells
        for r in range(gr):
            for c in range(gc):
                py = r * self.grid_res + self.grid_res // 2
                px = c * self.grid_res + self.grid_res // 2
                if py < h and px < w and drivable_mask[py, px] > 127:
                    grid[r, c] = 0

        # Mark obstacle cells
        if obstacles:
            for obs in obstacles:
                x1 = int(obs.get('x1', 0)) // self.grid_res
                y1 = int(obs.get('y1', 0)) // self.grid_res
                x2 = int(obs.get('x2', 0)) // self.grid_res
                y2 = int(obs.get('y2', 0)) // self.grid_res
                # Inflate by 1 cell for safety margin
                for r in range(max(0, y1 - 1), min(gr, y2 + 2)):
                    for c in range(max(0, x1 - 1), min(gc, x2 + 2)):
                        grid[r, c] = 1

        return grid

    def _nearest_free(self, grid, pos):
        """Find nearest free cell to pos via BFS."""
        gr, gc = grid.shape
        if (0 <= pos[0] < gr and 0 <= pos[1] < gc
                and grid[pos[0], pos[1]] == 0):
            return pos
        visited = set()
        queue = [pos]
        while queue:
            r, c = queue.pop(0)
            if (r, c) in visited:
                continue
            visited.add((r, c))
            if 0 <= r < gr and 0 <= c < gc and grid[r, c] == 0:
                return (r, c)
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < gr and 0 <= nc < gc:
                    queue.append((nr, nc))
        return None

    def _astar(self, grid, start, goal):
        """A* search on grid. Returns list of (row, col)."""
        gr, gc = grid.shape
        open_set = []
        heapq.heappush(open_set, (0, start))
        came_from = {}
        g_score = {start: 0}

        while open_set:
            _, current = heapq.heappop(open_set)
            if current == goal:
                return self._reconstruct(came_from, current)

            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1),
                           (-1, -1), (-1, 1), (1, -1), (1, 1)]:
                nr, nc = current[0] + dr, current[1] + dc
                neighbor = (nr, nc)
                if not (0 <= nr < gr and 0 <= nc < gc):
                    continue
                if grid[nr, nc] == 1:
                    continue
                move_cost = 1.414 if abs(dr) + abs(dc) == 2 else 1.0
                tent_g = g_score[current] + move_cost
                if tent_g < g_score.get(neighbor, float('inf')):
                    came_from[neighbor] = current
                    g_score[neighbor] = tent_g
                    h = abs(nr - goal[0]) + abs(nc - goal[1])
                    heapq.heappush(open_set, (tent_g + h, neighbor))

        return []  # no path found

    def _reconstruct(self, came_from, current):
        path = [current]
        while current in came_from:
            current = came_from[current]
            path.append(current)
        path.reverse()
        return path
