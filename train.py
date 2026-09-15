"""
训练主程序 —— 3D 智能小车 DQN 学习过程实时可视化。

运行后弹出两个面板：
- 左：3D 场景，蓝色小车实时追踪红色目标（蓝线是它本回合的轨迹）
- 右：学习曲线，每个回合的累计奖励逐渐爬升

用法：
    python train.py                     # 完整训练 + 实时可视化窗口
    python train.py --episodes 200      # 只训练 200 回合
    python train.py --no-render         # 不开窗口，快速训练并保存结果图
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
    parser = argparse.ArgumentParser(description="3D 智能小车 DQN 训练可视化")
    parser.add_argument("--episodes", type=int, default=500, help="训练回合数")
    parser.add_argument("--render-every", type=int, default=2,
                        help="每 N 个回合刷新一次画面（越大越快）")
    parser.add_argument("--save", type=str, default="training_result.png",
                        help="最终结果图保存路径")
    parser.add_argument("--no-render", action="store_true",
                        help="不打开窗口，只训练并保存结果图")
    args = parser.parse_args()

    # 不显示窗口时用无界面后端
    if args.no_render:
        matplotlib.use("Agg")

    env = CarEnv3D()
    agent = DQNAgent(state_dim=env._state().shape[0], n_actions=env.n_actions)

    # ---------------- 可视化画布 ----------------
    plt.ion()
    fig = plt.figure(figsize=(13, 6))
    ax3d = fig.add_subplot(121, projection="3d")
    ax2d = fig.add_subplot(122)

    # 左侧 3D 场景
    ax3d.set_xlim(-9, 9); ax3d.set_ylim(-9, 9); ax3d.set_zlim(-9, 9)
    ax3d.set_xlabel("X"); ax3d.set_ylabel("Y"); ax3d.set_zlabel("Z")
    ax3d.set_title("3D 智能小车 · DQN 学习追踪目标")
    goal_pt, = ax3d.plot([], [], [], "ro", markersize=9, label="目标")
    trail, = ax3d.plot([], [], [], "b-", alpha=0.6, lw=1.5, label="小车轨迹")
    car_pt, = ax3d.plot([], [], [], "bo", markersize=7, label="小车")
    ax3d.legend(loc="upper left")

    # 右侧学习曲线
    ax2d.set_title("每个回合累计奖励（学习曲线）")
    ax2d.set_xlabel("回合"); ax2d.set_ylabel("累计奖励")
    ax2d.grid(True, alpha=0.4)
    line_reward, = ax2d.plot([], [], "g-", lw=1.5)
    line_avg, = ax2d.plot([], [], "r--", lw=1.5, label="近50回合均值")
    ax2d.legend(loc="upper left")

    # ---------------- 训练循环 ----------------
    episode_rewards = []
    running_loss, loss_count = 0.0, 0

    for ep in range(1, args.episodes + 1):
        state = env.reset()
        total_reward = 0.0
        traj = [env.car_pos.copy()]
        done = False

        while not done:
            action = agent.act(state)
            state2, reward, done, _ = env.step(action)
            agent.remember(state, action, reward, state2, done)
            loss = agent.replay()
            if loss is not None:
                running_loss += loss
                loss_count += 1
            state = state2
            total_reward += reward
            traj.append(env.car_pos.copy())

        episode_rewards.append(total_reward)
        avg50 = float(np.mean(episode_rewards[-50:]))

        # ---- 更新可视化（仅渲染模式：每 render-every 回合刷新一次）----
        if not args.no_render and (ep % args.render_every == 0 or ep == 1):
            goal_pt.set_data([env.goal[0]], [env.goal[1]])
            goal_pt.set_3d_properties([env.goal[2]])

            traj_arr = np.array(traj)
            trail.set_data(traj_arr[:, 0], traj_arr[:, 1])
            trail.set_3d_properties(traj_arr[:, 2])

            car_pt.set_data([env.car_pos[0]], [env.car_pos[1]])
            car_pt.set_3d_properties([env.car_pos[2]])

            xs = list(range(1, len(episode_rewards) + 1))
            line_reward.set_data(xs, episode_rewards)
            if len(episode_rewards) >= 50:
                line_avg.set_data(xs, [np.mean(episode_rewards[max(0, i - 50):i + 1])
                                       for i in range(len(episode_rewards))])
            ax2d.relim(); ax2d.autoscale_view()

            fig.canvas.draw_idle()
            fig.canvas.flush_events()
            plt.pause(0.001)

        # ---- 打印进度 ----
        if ep % 20 == 0 or ep == 1:
            avg_loss = running_loss / max(loss_count, 1)
            print(f"回合 {ep:>4d} | 本回合奖励 {total_reward:6.1f} | "
                  f"近50回合均值 {avg50:6.1f} | 探索率 {agent.epsilon:.3f} | "
                  f"平均损失 {avg_loss:.4f}")
            running_loss, loss_count = 0.0, 0

    # ---------------- 收尾：补画最后一回合 3D + 完整曲线，再保存 ----------------
    goal_pt.set_data([env.goal[0]], [env.goal[1]])
    goal_pt.set_3d_properties([env.goal[2]])
    traj_arr = np.array(traj)
    trail.set_data(traj_arr[:, 0], traj_arr[:, 1])
    trail.set_3d_properties(traj_arr[:, 2])
    car_pt.set_data([env.car_pos[0]], [env.car_pos[1]])
    car_pt.set_3d_properties([env.car_pos[2]])

    xs = list(range(1, len(episode_rewards) + 1))
    line_reward.set_data(xs, episode_rewards)
    if len(episode_rewards) >= 50:
        line_avg.set_data(xs, [np.mean(episode_rewards[max(0, i - 50):i + 1])
                               for i in range(len(episode_rewards))])
    ax2d.relim(); ax2d.autoscale_view()
    fig.canvas.draw_idle()

    fig.savefig(args.save, dpi=120, bbox_inches="tight")
    print(f"\n训练完成！共 {args.episodes} 回合，结果图已保存为 {args.save}")

    if args.no_render:
        plt.close(fig)
    else:
        plt.ioff()
        plt.show()


if __name__ == "__main__":
    main()
