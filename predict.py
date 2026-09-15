"""
推理演示脚本 —— 加载训练好的模型，让小车"表演"它学会的技能。

完全不进行学习，只是利用已经训练好的 Q 网络，看小车如何追踪目标。
展示内容：
- 左：3D 场景，小车直接朝目标走去（轨迹收敛）
- 右：每个演示回合小车到目标的距离曲线（应一路下降）

用法：
    python predict.py                  # 加载 model.pkl 并 3D 可视化演示
    python predict.py --model my.pkl   # 指定模型文件
    python predict.py --no-render      # 不开窗口，只跑结果并保存图
"""
import argparse

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

# 中文字体（Windows 微软雅黑，避免图上中文显示为方块）
matplotlib.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "sans-serif"]
matplotlib.rcParams["axes.unicode_minus"] = False

from dqn_agent import DQNAgent
from environment import CarEnv3D


def main():
    parser = argparse.ArgumentParser(description="加载训练好的 3D 小车模型做推理演示")
    parser.add_argument("--model", type=str, default="model.pkl",
                        help="模型文件（由 train.py 生成）")
    parser.add_argument("--episodes", type=int, default=5, help="演示回合数")
    parser.add_argument("--save", type=str, default="predict_result.png",
                        help="结果图保存路径")
    parser.add_argument("--no-render", action="store_true",
                        help="不打开窗口，只跑结果并保存图")
    args = parser.parse_args()

    if args.no_render:
        matplotlib.use("Agg")

    env = CarEnv3D()
    agent = DQNAgent.load(args.model)
    agent.epsilon = 0.0   # 纯利用：完全按学到的策略走，不再随机
    print(f"已加载模型 {args.model}（学习步数 {agent.learn_steps}）")

    # ---------------- 画布 ----------------
    plt.ion()
    fig = plt.figure(figsize=(13, 6))
    ax3d = fig.add_subplot(121, projection="3d")
    ax2d = fig.add_subplot(122)

    ax3d.set_xlim(-9, 9); ax3d.set_ylim(-9, 9); ax3d.set_zlim(-9, 9)
    ax3d.set_xlabel("X"); ax3d.set_ylabel("Y"); ax3d.set_zlabel("Z")
    ax3d.set_title("推理演示 · 训练好的小车追踪目标")
    goal_pt, = ax3d.plot([], [], [], "ro", markersize=9, label="目标")
    trail, = ax3d.plot([], [], [], "b-", alpha=0.6, lw=1.5, label="小车轨迹")
    car_pt, = ax3d.plot([], [], [], "bo", markersize=7, label="小车")
    ax3d.legend(loc="upper left")

    ax2d.set_title("每个演示回合 · 小车到目标的距离")
    ax2d.set_xlabel("步数"); ax2d.set_ylabel("到目标距离")
    ax2d.grid(True, alpha=0.4)

    # ---------------- 推理循环 ----------------
    results = []            # 每个回合 (是否到达, 步数)
    last_dists = []         # 每个回合结束时距离
    all_distances = []      # 每个回合的完整距离序列（用于绘图）

    for ep in range(1, args.episodes + 1):
        state = env.reset()
        traj = [env.car_pos.copy()]
        dists = [np.linalg.norm(env.goal - env.car_pos)]
        reached = False
        done = False
        while not done:
            action = agent.act(state, training=False)   # 纯利用，不学习
            state, _, done, info = env.step(action)
            traj.append(env.car_pos.copy())
            dists.append(info["dist"])
            if info["dist"] < env.goal_radius:
                reached = True

        last_dists.append(dists[-1])
        all_distances.append(dists)
        results.append((reached, len(dists)))

        if not args.no_render:
            # 更新 3D 场景
            goal_pt.set_data([env.goal[0]], [env.goal[1]])
            goal_pt.set_3d_properties([env.goal[2]])
            traj_arr = np.array(traj)
            trail.set_data(traj_arr[:, 0], traj_arr[:, 1])
            trail.set_3d_properties(traj_arr[:, 2])
            car_pt.set_data([env.car_pos[0]], [env.car_pos[1]])
            car_pt.set_3d_properties([env.car_pos[2]])
            fig.canvas.draw_idle()
            fig.canvas.flush_events()
            plt.pause(0.4)

        status = "到达目标" if reached else "未到达（步数用尽）"
        print(f"回合 {ep}: {status} | 用时 {len(dists)} 步 | 结束距离 {dists[-1]:.2f}")

    # ---------------- 收尾：画出每个回合的距离曲线再保存 ----------------
    for dists in all_distances:
        ax2d.plot(list(range(len(dists))), dists, alpha=0.7, lw=1.5)
    ax2d.relim(); ax2d.autoscale_view()
    fig.canvas.draw_idle()
    fig.savefig(args.save, dpi=120, bbox_inches="tight")

    succ = sum(1 for r, _ in results if r)
    avg_dist = float(np.mean(last_dists))
    print(f"\n演示完成：{args.episodes} 回合中 {succ} 次到达目标"
          f"（成功率 {succ / args.episodes:.0%}）")
    print(f"平均结束距离 {avg_dist:.2f}（越小说明越接近目标）")
    print(f"结果图已保存为 {args.save}")

    if args.no_render:
        plt.close(fig)
    else:
        plt.ioff()
        plt.show()


if __name__ == "__main__":
    main()
