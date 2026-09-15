"""
从零手写的多层感知机（MLP）—— 深度学习的最小单元。

不依赖任何深度学习框架，只用 numpy 实现：
- 前向传播：输入 → 隐藏层(ReLU) → 输出层(线性)
- 反向传播：从输出误差逐层计算梯度
- 随机梯度下降(SGD)：沿负梯度更新权重

这个网络将作为 DQN 的"大脑"（Q 网络），把状态映射成每个动作的 Q 值。
"""
import numpy as np


class NeuralNet:
    def __init__(self, sizes: list, lr: float = 1e-3, seed: int | None = None):
        """
        sizes: 每层神经元数，例如 [6, 64, 64, 6] 表示 输入6 → 隐藏64 → 隐藏64 → 输出6
        lr:    学习率
        """
        self.sizes = sizes
        self.lr = lr
        self.n_layers = len(sizes) - 1          # 权重层数
        if seed is not None:
            np.random.seed(seed)

        # He 初始化（适合 ReLU），权重 ~ N(0, sqrt(2/输入维度))
        self.W = []                             # 权重矩阵
        self.b = []                             # 偏置向量
        for i in range(self.n_layers):
            scale = np.sqrt(2.0 / sizes[i])
            self.W.append(np.random.randn(sizes[i], sizes[i + 1]) * scale)
            self.b.append(np.zeros(sizes[i + 1]))

    # ---------------------------------------------------------------
    # 前向传播
    # ---------------------------------------------------------------
    def forward(self, x: np.ndarray) -> np.ndarray:
        """前向计算，保存每层激活值供反向传播使用。返回网络输出。"""
        self.activations = [np.asarray(x, dtype=float).reshape(-1)]
        for i in range(self.n_layers - 1):          # 隐藏层：线性 + ReLU
            z = self.activations[i] @ self.W[i] + self.b[i]
            self.activations.append(np.maximum(0, z))   # ReLU
        out = self.activations[-1] @ self.W[-1] + self.b[-1]  # 输出层线性
        self.activations.append(out)
        return out

    def predict(self, x: np.ndarray) -> np.ndarray:
        """对外预测接口（Q 值）。"""
        return self.forward(x)

    # ---------------------------------------------------------------
    # 反向传播 + 梯度下降（一步在线更新）
    # ---------------------------------------------------------------
    def backprop_and_update(self, x: np.ndarray, d_out: np.ndarray) -> float:
        """
        针对单个样本做一次梯度下降。
        d_out: 输出层的误差梯度向量（只有被选中动作的位置非零，其余为 0）
        返回:  本次的 MSE 损失
        """
        self.forward(x)                 # 前向，填充 activations
        loss = 0.5 * float(np.dot(d_out, d_out))

        # 第一遍：从输出层向输入层传播，先算好每一层的梯度（用旧权重）
        grad_Ws = [None] * self.n_layers
        grad_bs = [None] * self.n_layers
        delta = d_out.astype(float)
        for i in range(self.n_layers - 1, -1, -1):
            grad_Ws[i] = np.outer(self.activations[i], delta)   # 权重梯度
            grad_bs[i] = delta.copy()                           # 偏置梯度
            if i > 0:   # 把误差继续传播到上一层
                delta = (self.W[i] @ delta) * (self.activations[i] > 0)

        # 第二遍：统一更新所有参数（SGD）
        for i in range(self.n_layers):
            self.W[i] -= self.lr * grad_Ws[i]
            self.b[i] -= self.lr * grad_bs[i]

        return loss
