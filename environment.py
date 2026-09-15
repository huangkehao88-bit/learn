"""
3D 小车环境 —— 深度强化学习(DQN)的"物理世界"。

小车在一个立方体空间里，学习如何追踪一个随机放置的目标。
- 状态：小车到目标的归一化方向向量 + 小车速度 + 归一化距离
- 动作：6 个离散动作（沿 x/y/z 三个轴的正负方向施加推力）
- 奖励：靠近目标得正奖励，远离得负奖励，到达目标给大奖励
"""
import numpy as np


class CarEnv3D:
    def __init__(self, world_size: float = 8.0, max_steps: int = 120,
                 goal_radius: float = 0.8):
        # world_size: 立方体世界半边长（世界范围 -world_size ~ +world_size）
        self.world_size = world_size
        self.max_steps = max_steps      # 每个回合最多走多少步
        self.goal_radius = goal_radius  # 距离目标多近算"到达"
        self.n_actions = 6              # +x -x +y -y +z -z
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

        dist = np.linalg.norm(self.goal - self.car_pos)
        # 奖励 = 本步距离的缩小量（推进奖励），越靠近得分越高
        reward = (self._prev_dist - dist) * 2.0
        done = False

        if dist < self.goal_radius:      # 到达目标：大奖励，回合结束
            reward += 20.0
            done = True
        elif self.step_count >= self.max_steps:  # 步数耗尽，回合结束
            done = True

        self._prev_dist = dist
        return self._state(), reward, done, {"dist": dist}

    # ---------------------------------------------------------------
    # 内部实现
    # ---------------------------------------------------------------
    def _random_goal(self):
        """在世界内随机放一个目标，但不要离小车太近。"""
        goal = np.random.uniform(-self.world_size, self.world_size, size=3)
        while np.linalg.norm(goal) < 3.0:  # 避免目标几乎贴着起点
            goal = np.random.uniform(-self.world_size, self.world_size, size=3)
        return goal

    def _action_to_thrust(self, action: int) -> np.ndarray:
        """把 0~5 的动作编号映射为三维推力方向。"""
        thrust = np.zeros(3)
        axis = action // 2          # 0:x, 1:y, 2:z
        sign = 1.0 if action % 2 == 0 else -1.0
        thrust[axis] = sign * 1.5
        return thrust

    def _state(self) -> np.ndarray:
        """状态特征向量：方向 + 速度 + 距离，全部归一化到合理范围。"""
        rel = self.goal - self.car_pos
        dist = np.linalg.norm(rel)
        direction = rel / (dist + 1e-6)             # 指向目标的单位向量
        norm_dist = dist / (self.world_size * 2.0)  # 归一化距离(0~1)
        norm_vel = self.car_vel / 2.0               # 归一化速度(-1~1)
        return np.concatenate([direction, norm_vel, [norm_dist]])
