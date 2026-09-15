"""
拟真城市道路环境 —— 不规则扭曲路网 + 稀疏街区。

真实城市特点：
- 每个道路交叉口在网格基础上随机扭曲，街道被掰弯、错位
- 出现斜路、错位路口、长短不一的街区，不再是规整棋盘
- 小车只能沿道路（交叉口之间的街道）行驶，不能抄近道穿楼
- 楼稀疏分布（部分街区留作空地/公园），高低不一

状态（5 维）：到目标单位方向(2) + 归一化距离(1) + 归一化位置(2)
动作（4）：上(沿-z) 下(沿+z) 左(沿-x) 右(沿+x) —— 在路网拓扑上移动
奖励：靠近目标 +1 / 远离 -1 / 撞边界(走不动) -0.5 / 到达目标 +20
"""
import numpy as np


class CarEnv3D:
    def __init__(self, grid=6, spacing=2.0, max_steps=100):
        self.grid = grid
        self.spacing = spacing
        self.max_steps = max_steps
        self.n_actions = 4
        rng = np.random.default_rng(42)          # 固定种子：每次运行同一张城市路网

        # 道路交叉口：基础网格 + 随机扭曲 -> 不规则街道
        self.node_xy = np.zeros((grid, grid, 2))
        for i in range(grid):
            for j in range(grid):
                ox = rng.uniform(-0.32, 0.32) * spacing
                oz = rng.uniform(-0.32, 0.32) * spacing
                self.node_xy[i, j] = [i * spacing + ox, j * spacing + oz]

        # 道路边：相邻交叉口之间的街道（直线，因为节点扭曲所以不规整）
        self.road_edges = []
        for i in range(grid):
            for j in range(grid - 1):            # 竖直街道
                self.road_edges.append((tuple(self.node_xy[i, j]),
                                        tuple(self.node_xy[i, j + 1])))
        for i in range(grid - 1):
            for j in range(grid):                # 水平街道
                self.road_edges.append((tuple(self.node_xy[i, j]),
                                        tuple(self.node_xy[i + 1, j])))

        # 城市范围（供画布）
        xs = self.node_xy[:, :, 0]; zs = self.node_xy[:, :, 1]
        self.world_bounds = (xs.min() - 0.7, xs.max() + 0.7,
                             zs.min() - 0.7, zs.max() + 0.7)

        self._build_buildings(rng)
        self.steps = 0
        self.car = (0, 0)
        self.goal = (grid - 1, grid - 1)
        self.reset()

    # ---------------- 坐标换算 ----------------
    def _world(self, cell):
        """交叉口索引 -> 世界坐标 (x, z)。"""
        ix, iz = cell
        return float(self.node_xy[ix, iz, 0]), float(self.node_xy[ix, iz, 1])

    def car_world(self):
        return self._world(self.car)

    def goal_world(self):
        return self._world(self.goal)

    def _dist(self, a, b):
        """两个交叉口之间的欧氏距离（统计用）。"""
        xa, za = self._world(a)
        xb, zb = self._world(b)
        return float(np.hypot(xb - xa, zb - za))

    # ---------------- 建筑（稀疏：部分街区留空）----------------
    def _build_buildings(self, rng):
        self.buildings = []
        for i in range(self.grid - 1):
            for j in range(self.grid - 1):
                p00, p10 = self.node_xy[i, j], self.node_xy[i + 1, j]
                p01, p11 = self.node_xy[i, j + 1], self.node_xy[i + 1, j + 1]
                cx = (p00[0] + p10[0] + p01[0] + p11[0]) / 4
                cz = (p00[1] + p10[1] + p01[1] + p11[1]) / 4
                sx = np.hypot(p10[0] - p00[0], p10[1] - p00[1]) * 0.55
                sz = np.hypot(p01[0] - p00[0], p01[1] - p00[1]) * 0.55
                if rng.random() < 0.3:            # 30% 街区留空（空地/公园）
                    continue
                h = rng.uniform(1.0, 2.6)
                self.buildings.append((cx, cz, sx, sz, h))

    # ---------------- 环境接口 ----------------
    def _state(self):
        cx, cz = self.car_world()
        gx, gz = self.goal_world()
        dx, dz = gx - cx, gz - cz
        dist = np.hypot(dx, dz)
        unit = np.array([dx, dz]) / max(dist, 1e-6)
        return np.concatenate([unit, [dist / 10.0, cx / 10.0, cz / 10.0]])

    def reset(self):
        self.car = (0, 0)
        choices = [(i, j) for i in range(self.grid)
                   for j in range(self.grid) if (i, j) != (0, 0)]
        self.goal = choices[np.random.randint(len(choices))]
        self.steps = 0
        return self._state()

    def step(self, action):
        self.steps += 1
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
        self.car = (ix, iz)

        d0 = self._dist(old, self.goal)
        d1 = self._dist(self.car, self.goal)
        if self.car == self.goal:
            reward, done = 20.0, True
        elif not moved:
            reward, done = -0.5, False
        else:
            reward = 1.0 if d1 < d0 else -1.0
            done = False
        if self.steps >= self.max_steps:
            done = True
        return self._state(), reward, done, {"dist": d1}
