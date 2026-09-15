"""
拟真城市道路环境 —— 不规则街道交错布局。

真实城市特点：
- 街道不等距、长短不一，街区大小不等（不是规整棋盘）
- 竖直街道（沿 x）与水平街道（沿 z）交错，形成大小不同的街区
- 小车只能在街道交叉点之间的道路上行驶，不能穿过街区（楼）
- 目标随机选一个交叉点，小车要沿道路找到最优路径

状态（5 维）：到目标单位方向(2) + 归一化距离(1) + 归一化位置(2)
动作（4）：上(沿-z) 下(沿+z) 左(沿-x) 右(沿+x)
奖励：靠近目标 +1 / 远离 -1 / 撞边界(走不动) -0.5 / 到达目标 +20
"""
import numpy as np


class CarEnv3D:
    def __init__(self, max_steps=100):
        self.max_steps = max_steps
        # 不规则城市街道位置：竖直街道（沿 x）和水平街道（沿 z）
        # 间距不等、街区大小不一，交错成真实城市路网
        self.street_x = np.array([0.0, 1.6, 3.3, 5.0, 6.8, 8.6, 10.0])
        self.street_z = np.array([0.0, 1.4, 3.1, 4.9, 7.0, 8.8, 10.0])
        self.nx = len(self.street_x)          # 竖直街道条数
        self.nz = len(self.street_z)          # 水平街道条数
        self.n_actions = 4
        self.steps = 0
        self.car = (0, 0)                     # 当前交叉点索引 (ix, iz)
        self.goal = (self.nx - 1, self.nz - 1)
        self._build_buildings()
        self.reset()

    # ---------------- 坐标换算 ----------------
    def _world(self, cell):
        """交叉点索引 -> 世界坐标 (x, z)。"""
        ix, iz = cell
        return float(self.street_x[ix]), float(self.street_z[iz])

    def car_world(self):
        return self._world(self.car)

    def goal_world(self):
        return self._world(self.goal)

    def _dist(self, a, b):
        """两个交叉点之间的欧氏距离（统计用）。"""
        xa, za = self._world(a)
        xb, zb = self._world(b)
        return float(np.hypot(xb - xa, zb - za))

    # ---------------- 建筑（每个街区一栋楼，大小随街区）----------------
    def _build_buildings(self):
        self.buildings = []
        for i in range(self.nx - 1):
            for j in range(self.nz - 1):
                x0, x1 = self.street_x[i], self.street_x[i + 1]
                z0, z1 = self.street_z[j], self.street_z[j + 1]
                cx, cz = (x0 + x1) / 2, (z0 + z1) / 2
                sx = (x1 - x0) * 0.72          # 楼占街区约 72%，留出道路
                sz = (z1 - z0) * 0.72
                h = np.random.uniform(1.2, 3.4)  # 楼高不一
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
        choices = [(i, j) for i in range(self.nx)
                   for j in range(self.nz) if (i, j) != (0, 0)]
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
        elif action == 1 and iz < self.nz - 1:
            iz += 1
        elif action == 2 and ix > 0:
            ix -= 1
        elif action == 3 and ix < self.nx - 1:
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
