"""
DQN 智能体 —— 强化学习的"大脑"。

它利用手写的神经网络（Q 网络），教会小车"看到什么状态，走哪个动作"。
核心组件：
- ε-greedy 探索：一开始多随机尝试，慢慢转为利用经验
- 经验回放：把过去的经验存起来随机抽样学习，打破数据相关性
- 目标网络：提供一个"稳定的标准"来算目标值，让学习更平稳
- 模型保存/加载：训练成果可复用，不必每次重新训练
"""
import pickle
import random
from collections import deque

import numpy as np

from neural_net import NeuralNet


class DQNAgent:
    def __init__(self, state_dim: int, n_actions: int, lr: float = 5e-4,
                 gamma: float = 0.96, epsilon: float = 1.0,
                 epsilon_min: float = 0.05, epsilon_decay: float = 0.99,
                 buffer_size: int = 20000, batch_size: int = 32,
                 target_update_interval: int = 60, seed: int | None = None):
        self.state_dim = state_dim
        self.n_actions = n_actions
        self.gamma = gamma                      # 折扣因子：未来奖励的打折程度
        self.epsilon = epsilon                  # 当前探索率
        self.epsilon_min = epsilon_min          # 探索率下限
        self.epsilon_decay = epsilon_decay      # 每次学习后探索率衰减比例
        self.batch_size = batch_size
        self.target_update_interval = target_update_interval
        self.learn_steps = 0
        if seed is not None:
            random.seed(seed)

        # 两个结构相同的网络：Q 网络(实时更新) + 目标网络(定期同步)
        sizes = [state_dim, 64, 64, n_actions]
        self.q_net = NeuralNet(sizes, lr=lr)
        self.target_net = NeuralNet(sizes, lr=lr)
        self._sync_target()

        # 经验回放缓冲（先进先出，满了丢弃最旧的）
        self.memory = deque(maxlen=buffer_size)

    # ---------------------------------------------------------------
    def act(self, state, training: bool = True) -> int:
        """ε-greedy：按探索率随机，否则选 Q 值最大的动作。"""
        if training and np.random.rand() < self.epsilon:
            return np.random.randint(self.n_actions)
        q = self.q_net.predict(state)
        return int(np.argmax(q))

    def remember(self, s, a, r, s2, done):
        """把一次经验 (状态, 动作, 奖励, 新状态, 是否结束) 存入记忆。"""
        self.memory.append((s, a, r, s2, done))

    def replay(self):
        """从记忆中采样一批经验学习，返回平均损失（不足一批则返回 None）。"""
        if len(self.memory) < self.batch_size:
            return None

        batch = random.sample(self.memory, self.batch_size)
        total_loss = 0.0
        for s, a, r, s2, done in batch:
            # Double DQN：用 Q 网络挑最优动作，再用目标网络评估该动作的值
            # （减少 Q 值过估计，比普通 DQN 学习更稳定）
            best_action = int(np.argmax(self.q_net.predict(s2)))
            q_next = self.target_net.predict(s2)[best_action]
            target = r if done else r + self.gamma * float(q_next)
            # 裁剪目标值，防止 Q 值发散（DQN 稳定的关键技巧）
            target = float(np.clip(target, -30.0, 30.0))
            # 只让"被选动作"的 Q 值朝目标靠近，其余动作的误差置 0
            q = self.q_net.predict(s)
            d_out = np.zeros_like(q)
            d_out[a] = q[a] - target
            total_loss += self.q_net.backprop_and_update(s, d_out)

        # 定期同步目标网络权重
        self.learn_steps += 1
        if self.learn_steps % self.target_update_interval == 0:
            self._sync_target()

        # 探索率衰减（越到后期越"经验主义"，少随机）
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        return total_loss / len(batch)

    # ---------------------------------------------------------------
    def _sync_target(self):
        """把 Q 网络的权重整体复制给目标网络。"""
        for i in range(len(self.q_net.W)):
            self.target_net.W[i] = self.q_net.W[i].copy()
            self.target_net.b[i] = self.q_net.b[i].copy()

    # ---------------------------------------------------------------
    # 模型保存 / 加载（权重序列化，训练成果可复用）
    # ---------------------------------------------------------------
    def save(self, path: str):
        """把 Q 网络权重和超参数保存到文件（pickle 序列化）。"""
        data = {
            "state_dim": self.state_dim,
            "n_actions": self.n_actions,
            "gamma": self.gamma,
            "epsilon": self.epsilon,
            "epsilon_min": self.epsilon_min,
            "epsilon_decay": self.epsilon_decay,
            "learn_steps": self.learn_steps,
            "q_W": self.q_net.W,     # 每层权重矩阵
            "q_b": self.q_net.b,     # 每层偏置向量
        }
        with open(path, "wb") as f:
            pickle.dump(data, f)

    @classmethod
    def load(cls, path: str, lr: float = 5e-4):
        """从文件加载模型，返回一个权重已恢复的 DQNAgent。

        加载后保留保存时的超参数；推理时建议把探索率设为 0 做纯利用。
        """
        with open(path, "rb") as f:
            data = pickle.load(f)

        agent = cls(
            state_dim=data["state_dim"],
            n_actions=data["n_actions"],
            lr=lr,
            gamma=data["gamma"],
            epsilon=data["epsilon"],
            epsilon_min=data["epsilon_min"],
            epsilon_decay=data["epsilon_decay"],
        )
        agent.learn_steps = data["learn_steps"]

        # 恢复 Q 网络权重（结构需匹配：层数、每层维度一致）
        if len(agent.q_net.W) != len(data["q_W"]):
            raise ValueError("模型结构与当前配置不匹配，无法加载")
        for i in range(len(agent.q_net.W)):
            if agent.q_net.W[i].shape != data["q_W"][i].shape:
                raise ValueError(f"第 {i} 层权重形状不匹配")
            agent.q_net.W[i] = data["q_W"][i]
            agent.q_net.b[i] = data["q_b"][i]
        agent._sync_target()
        return agent
