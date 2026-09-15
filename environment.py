"""
拟真城市道路环境 —— 小车在城市里只能沿道路行驶，学习到达目标的最优路径。

城市是 x-z 平面上的道路网格：横向/纵向道路把城市分成街区，街区里是高楼。
小车是一个玩具小车，只能沿道路格点移动（不能穿楼、不能离开道路），
目标是到达某个道路格点。

- 状态：到目标的单位方向 + 归一化距离 + 归一化当前位置
- 动作：4 个（上 / 下 / 左 / 右）
- 奖励：靠近目标 +，到达目标 +20，无效移动(撞边界) -
"""
import numpy as np


class CarEnv3D:
    """城市道路网格环境（俯视即道路网，街区是楼）。"""

    def __init__(self, grid: int = 6, spacing: float = 2.0,
                 max_steps: int = 80, seed: int | None = None):
        self.grid = grid            # grid × grid 个道路格点
        self.spacing = spacing      # 相邻格点间距（世界单位）
        self.max_steps = max_steps  # 每个回合最多走多少步
        self.n_actions = 4          # 0上 1下 2左 3右
        if seed is not None:
            np.random.seed(seed)

        # 楼：每个街区一栋（中心x, 中心z, 长, 宽, 高）
        self.buildings = self._make_buildings()
        self.reset()

    # ---------------------------------------------------------------
    # 环境主接口
    # ---------------------------------------------------------------
    def reset(self):
        """小车从 (0,0) 出发，目标随机放在某条道路格点，返回初始状态。"""
        self.car = (0, 0)
        self.goal = (np.random.randint(self.grid), np.random.randint(self.grid))
        while self.goal == self.car:            # 目标不能就是起点
            self.goal = (np.random.randint(self.grid), np.random.randint(self.grid))
        self.step_count = 0
        self._prev_car = self.car
        return self._state()

    def step(self, action: int):
        """沿道路移动一格，返回 (新状态, 奖励, 是否结束, 附加信息)。"""
        i, j = self.car
        move = {0: (0, -1), 1: (0, 1), 2: (-1, 0), 3: (1, 0)}[action]
        ni, nj = i + move[0], j + move[1]
        valid = 0 <= ni < self.grid and 0 <= nj < self.grid
        if valid:
            self.car = (ni, nj)
        self.step_count += 1

        dist = self._dist(self.car, self.goal)
        prev = self._dist(self._prev_car, self.goal)
        reward = float(prev - dist)     # 靠近目标得 +1，远离得 -1
        done = False
        if not valid:
            reward -= 0.5               # 撞到道路边界（无效移动）惩罚
        if self.car == self.goal:       # 到达目标
            reward += 20.0
            done = True
        elif self.step_count >= self.max_steps:
            done = True

        self._prev_car = self.car
        return self._state(), reward, done, {"dist": dist, "car": self.car}

    def car_world(self):
        """小车当前格点对应的世界坐标 (x, z)。"""
        return self.car[0] * self.spacing, self.car[1] * self.spacing

    def goal_world(self):
        """目标格点对应的世界坐标 (x, z)。"""
        return self.goal[0] * self.spacing, self.goal[1] * self.spacing

    # ---------------------------------------------------------------
    # 内部实现
    # ---------------------------------------------------------------
    def _make_buildings(self):
        """在每个街区内生成一栋随机高度的楼。"""
        buildings = []
        for i in range(self.grid - 1):
            for j in range(self.grid - 1):
                cx = (i + 0.5) * self.spacing
                cz = (j + 0.5) * self.spacing
                height = float(np.random.uniform(2.0, 4.5))
                size = self.spacing * 0.7       # 楼几乎占满街区，留出道路
                buildings.append((cx, cz, size, size, height))
        return buildings

    @staticmethod
    def _dist(a, b):
        """曼哈顿距离（贴合道路网格）。"""
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def _state(self) -> np.ndarray:
        """状态：到目标方向(2) + 归一化距离(1) + 归一化位置(2)。"""
        dg = np.array(self.goal, dtype=float) - np.array(self.car, dtype=float)
        norm = np.linalg.norm(dg) + 1e-6
        direction = dg / norm
        dist = self._dist(self.car, self.goal) / (2.0 * (self.grid - 1))
        pos = np.array(self.car, dtype=float) / (self.grid - 1)
        return np.concatenate([direction, [dist], pos])
