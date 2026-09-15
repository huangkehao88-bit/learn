"""
拟真城市自动驾驶环境 —— 红绿灯 + 让行规则 + 路上其他车（会车）。

城市特点：
- 不规则扭曲路网，交叉口错落、街区大小不一
- 若干内部路口装有红绿灯（红/绿周期切换），红灯必须等待
- 3 辆其他车沿固定环线行驶，小车前方有车时必须让行等待，避免碰撞
- 小车只能沿道路行驶，到达目标为任务

状态（7 维）：到目标单位方向(2) + 归一化距离(1) + 归一化位置(2)
          + 前方是否有车/红灯(1，1=需让行/等灯) + 前方是否被NPC车占用(1)
动作（4）：上(沿-z) 下(沿+z) 左(沿-x) 右(沿+x)
奖励：靠近 +1 / 远离 -1 / 撞边界 -0.5 / 等红灯 -0.3 / 前方有车让行 -6 / 碰撞 -6 / 到达 +20
"""
import numpy as np


class CarEnv3D:
    def __init__(self, grid=6, spacing=2.0, max_steps=120):
        self.grid = grid
        self.spacing = spacing
        self.max_steps = max_steps
        self.n_actions = 4
        rng = np.random.default_rng(42)          # 固定种子：同一张城市路网

        # ---- 交叉口：基础网格 + 随机扭曲（不规则道路）----
        self.node_xy = np.zeros((grid, grid, 2))
        for i in range(grid):
            for j in range(grid):
                ox = rng.uniform(-0.45, 0.45) * spacing
                oz = rng.uniform(-0.45, 0.45) * spacing
                self.node_xy[i, j] = [i * spacing + ox, j * spacing + oz]
        self.road_edges = []
        for i in range(grid):
            for j in range(grid - 1):
                self.road_edges.append((tuple(self.node_xy[i, j]),
                                        tuple(self.node_xy[i, j + 1])))
        for i in range(grid - 1):
            for j in range(grid):
                self.road_edges.append((tuple(self.node_xy[i, j]),
                                        tuple(self.node_xy[i + 1, j])))
        xs = self.node_xy[:, :, 0]; zs = self.node_xy[:, :, 1]
        self.world_bounds = (xs.min() - 0.7, xs.max() + 0.7,
                             zs.min() - 0.7, zs.max() + 0.7)
        self._build_buildings(rng)

        # ---- 红绿灯：若干内部路口 ----
        self.signal_nodes = [(1, 1), (2, 3), (3, 2), (4, 4)]
        self.signal_phase = {n: int(rng.integers(0, 2)) for n in self.signal_nodes}
        self.signal_half_period = 6              # 每 6 步红绿切换一次

        # ---- 其他车：沿环线行驶 ----
        self.npc_routes = [
            [(0, 0), (0, 1), (0, 2), (0, 3), (0, 4), (0, 5), (1, 5), (2, 5),
             (3, 5), (4, 5), (5, 5), (5, 4), (5, 3), (5, 2), (5, 1), (5, 0),
             (4, 0), (3, 0), (2, 0), (1, 0)],
            [(1, 1), (1, 2), (1, 3), (1, 4), (2, 4), (3, 4), (4, 4),
             (4, 3), (4, 2), (4, 1), (3, 1), (2, 1)],
            [(1, 2), (1, 3), (1, 4), (1, 5), (2, 5), (3, 5), (3, 4),
             (3, 3), (3, 2), (2, 2)],
        ]
        self.npc_cars = []
        for route in self.npc_routes:
            start = int(rng.integers(0, len(route)))
            if route[start] == (0, 0):
                start = (start + 3) % len(route)
            self.npc_cars.append({'route': route, 'idx': start, 'pos': route[start]})

        self.steps = 0
        self.car = (0, 0)
        self.goal = (grid - 1, grid - 1)
        self.reset()

    # ---------------- 坐标换算 ----------------
    def _world(self, cell):
        ix, iz = cell
        return float(self.node_xy[ix, iz, 0]), float(self.node_xy[ix, iz, 1])

    def car_world(self):
        return self._world(self.car)

    def goal_world(self):
        return self._world(self.goal)

    def _dist(self, a, b):
        xa, za = self._world(a)
        xb, zb = self._world(b)
        return float(np.hypot(xb - xa, zb - za))

    # ---------------- 建筑（稀疏街区）----------------
    def _build_buildings(self, rng):
        self.buildings = []
        for i in range(self.grid - 1):
            for j in range(self.grid - 1):
                p00, p10 = self.node_xy[i, j], self.node_xy[i + 1, j]
                p01, p11 = self.node_xy[i, j + 1], self.node_xy[i + 1, j + 1]
                cx = (p00[0] + p10[0] + p01[0] + p11[0]) / 4
                cz = (p00[1] + p10[1] + p01[1] + p11[1]) / 4
                sx = np.hypot(p10[0] - p00[0], p10[1] - p00[1]) * 0.40
                sz = np.hypot(p01[0] - p00[0], p01[1] - p00[1]) * 0.40
                if rng.random() < 0.4:            # 40% 留空
                    continue
                h = rng.uniform(1.0, 2.6)
                self.buildings.append((cx, cz, sx, sz, h))

    # ---------------- 红绿灯 / 其他车 ----------------
    def _signal_is_red(self, node):
        if node not in self.signal_nodes:
            return False
        return ((self.steps // self.signal_half_period) + self.signal_phase[node]) % 2 == 1

    def _npc_positions(self):
        return {npc['pos'] for npc in self.npc_cars}

    def _move_npcs(self):
        for npc in self.npc_cars:
            npc['idx'] = (npc['idx'] + 1) % len(npc['route'])
            npc['pos'] = npc['route'][npc['idx']]

    def _front_cell(self):
        """朝目标方向（使距离减小）的下一格。"""
        ix, iz = self.car
        best, best_d = self.car, self._dist(self.car, self.goal)
        for ni, nj in [(ix, iz - 1), (ix, iz + 1), (ix - 1, iz), (ix + 1, iz)]:
            if 0 <= ni < self.grid and 0 <= nj < self.grid:
                d = self._dist((ni, nj), self.goal)
                if d < best_d:
                    best_d, best = d, (ni, nj)
        return best

    def _front_blocked(self):
        fc = self._front_cell()
        if fc in self._npc_positions():
            return 1.0
        if self._signal_is_red(fc):
            return 1.0
        return 0.0

    # ---------------- 环境接口 ----------------
    def _state(self):
        cx, cz = self.car_world()
        gx, gz = self.goal_world()
        dx, dz = gx - cx, gz - cz
        dist = np.hypot(dx, dz)
        unit = np.array([dx, dz]) / max(dist, 1e-6)
        fc = self._front_cell()
        front_car = 1.0 if fc in self._npc_positions() else 0.0
        front_signal = 1.0 if self._signal_is_red(fc) else 0.0
        return np.concatenate([unit, [dist / 10.0, cx / 10.0, cz / 10.0],
                               [front_car, front_signal]])

    def reset(self):
        self.car = (0, 0)
        choices = [(i, j) for i in range(self.grid)
                   for j in range(self.grid) if (i, j) != (0, 0)]
        self.goal = choices[np.random.randint(len(choices))]
        self.steps = 0
        for npc in self.npc_cars:                # 其他车回到路线起点
            npc['idx'] = 0
            npc['pos'] = npc['route'][0]
        return self._state()

    def step(self, action):
        self.steps += 1
        self._move_npcs()                        # 其他车先移动
        hit = (self.car in self._npc_positions())  # 是否被其他车撞到

        ix, iz = self.car
        old = self.car
        moved = True
        if action == 0 and iz > 0:
            iz -= 1
        elif action == 1 and iz < self.grid - 1:
            iz += 1
        elif action == 2 and ix > 0:
            ix -= 1
        elif action == 3 and ix < self.grid - 1:
            ix += 1
        else:
            moved = False
        new = (ix, iz)

        if moved and new in self._npc_positions():
            self.car = old                      # 前方有车，让行：停在原地
            reward, done = -6.0, False
        elif moved and self._signal_is_red(new):
            self.car = old                      # 红灯，等待
            reward, done = -0.3, False
        else:
            self.car = new
            if self.car == self.goal:
                reward, done = 20.0, True
            else:
                d0 = self._dist(old, self.goal)
                d1 = self._dist(new, self.goal)
                reward = 1.0 if d1 < d0 else -1.0
                done = False

        if hit:                                 # 被其他车撞到（即使让行也可能被追上）
            reward, done = -6.0, False
        if self.steps >= self.max_steps:
            done = True
        return self._state(), reward, done, {"dist": self._dist(self.car, self.goal)}
