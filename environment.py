"""
3D 智能小车环境（带建筑物） —— 深度强化学习(DQN)的"物理世界"。

小车在一个立方体空间里，追踪一个随机放置的目标，同时要避开固定位置的建筑物。
- 状态：指向目标的单位方向 + 小车速度 + 归一化目标距离
         + 感应周围每个建筑物的归一化距离（"距离传感器"）
- 动作：6 个离散动作（沿 x/y/z 三个轴的正负方向施加推力）
- 奖励：靠近目标得正奖励，撞到建筑物得负惩罚，到达目标给大奖励
"""
import numpy as np


class CarEnv3D:
    def __init__(self, world_size: float = 8.0, max_steps: int = 150,
                 goal_radius: float = 0.8, car_radius: float = 0.3,
                 n_obstacles: int = 4, obstacle_radius: float = 1.2):
        self.world_size = world_size      # 立方体世界半边长（-8 ~ +8）
        self.max_steps = max_steps        # 每个回合最多走多少步
        self.goal_radius = goal_radius    # 距离目标多近算"到达"
        self.car_radius = car_radius      # 小车的碰撞半径
        self.n_actions = 6                # +x -x +y -y +z -z

        # 固定布局的建筑物（中心点, 半径），方便可视化与学习
        self.obstacles = self._make_obstacles(n_obstacles, obstacle_radius)
        self.reset()

    # ---------------------------------------------------------------
    # 环境主接口（reset / step / state），供 DQN 使用
    # ---------------------------------------------------------------
    def reset(self):
        """开始新回合：小车回到原点，目标随机放置，返回初始状态。"""
        self.car_pos = np.zeros(3)                      # 小车位置
        self.car_vel = np.zeros(3)                      # 小车速度
        self.goal = self._random_goal()
        self.step_count = 0
        self._prev_dist = np.linalg.norm(self.goal - self.car_pos)
        return self._state()

    def step(self, action: int):
        """执行一个动作，返回 (新状态, 奖励, 是否结束, 附加信息)。"""
        thrust = self._action_to_thrust(action)
        # 简化牛顿第二定律：推力改变速度，速度改变位置
        self.car_vel += thrust * 0.1
        self.car_vel = np.clip(self.car_vel, -2.0, 2.0)  # 限速
        self.car_pos += self.car_vel * 0.1
        # 世界边界（碰到边界就停在边界上）
        self.car_pos = np.clip(self.car_pos, -self.world_size, self.world_size)
        self.step_count += 1

        # ---- 与建筑物的碰撞检测：撞上就反弹开并扣分 ----
        collision = False
        for center, radius in self.obstacles:
            if np.linalg.norm(self.car_pos - center) < radius + self.car_radius:
                collision = True
                # 把小车推出障碍物表面（反弹效果）
                outward = self.car_pos - center
                dist_to_c = np.linalg.norm(outward)
                if dist_to_c > 1e-6:
                    self.car_pos = center + outward / dist_to_c * (radius + self.car_radius)
                break

        dist = np.linalg.norm(self.goal - self.car_pos)
        # 奖励 = 本步距离的缩小量（推进奖励）
        reward = (self._prev_dist - dist) * 2.0
        done = False

        if collision:                       # 撞建筑物：惩罚
            reward -= 5.0
        if dist < self.goal_radius:         # 到达目标：大奖励，回合结束
            reward += 20.0
            done = True
        elif self.step_count >= self.max_steps:   # 步数耗尽，回合结束
            done = True

        self._prev_dist = dist
        return self._state(), reward, done, {"dist": dist, "collision": collision}

    # ---------------------------------------------------------------
    # 内部实现
    # ---------------------------------------------------------------
    def _make_obstacles(self, n: int, radius: float):
        """生成固定位置的建筑物布局（避开起点原点附近）。"""
        layout = [
            (np.array([3.5,  1.5,  1.0]), radius),
            (np.array([-2.5,  3.0, -1.5]), radius),
            (np.array([1.0, -3.5,  2.0]), radius),
            (np.array([-1.0,  0.5, -3.5]), radius),
        ]
        # 按需取前 n 个
        return [(layout[i][0], layout[i][1]) for i in range(min(n, len(layout)))]

    def _random_goal(self):
        """在世界内随机放一个目标：不贴起点，且不在任何建筑物内部。"""
        while True:
            goal = np.random.uniform(-self.world_size, self.world_size, size=3)
            if np.linalg.norm(goal) < 3.0:          # 避免目标贴着起点
                continue
            # 目标不能落在建筑物内（否则永远无法到达）
            if all(np.linalg.norm(goal - c) > r + 0.3 for c, r in self.obstacles):
                return goal

    def _action_to_thrust(self, action: int) -> np.ndarray:
        """把 0~5 的动作编号映射为三维推力方向。"""
        thrust = np.zeros(3)
        axis = action // 2          # 0:x, 1:y, 2:z
        sign = 1.0 if action % 2 == 0 else -1.0
        thrust[axis] = sign * 1.5
        return thrust

    def _state(self) -> np.ndarray:
        """状态特征向量：方向 + 速度 + 目标距离 + 各建筑物感应距离。"""
        rel = self.goal - self.car_pos
        dist = np.linalg.norm(rel)
        direction = rel / (dist + 1e-6)             # 指向目标的单位向量
        norm_dist = dist / (self.world_size * 2.0)  # 归一化目标距离(0~1)
        norm_vel = self.car_vel / 2.0               # 归一化速度(-1~1)

        # 距离传感器：感应每个建筑物的归一化距离（越小越近）
        obs_dist = []
        for center, radius in self.obstacles:
            surface_dist = np.linalg.norm(self.car_pos - center) - radius
            obs_dist.append(surface_dist / (self.world_size * 2.0))

        return np.concatenate([direction, norm_vel, [norm_dist], obs_dist])
