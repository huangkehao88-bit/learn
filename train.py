"""
训练主程序 —— 拟真城市 DQN（Double DQN）城市导航学习。

运行后弹出实时 3D 城市界面：
- 左：拟真城市（灰色道路 + 白色路中线 + 高楼街区 + 玩具小车沿道路行驶 + 绿色路径）
- 右：学习曲线（每个回合累计奖励，逐渐爬升）

小车必须沿道路格点行驶，不能穿楼，DQN 学会找到到达目标的最优路径。

用法：
    python train.py                      # 训练 + 实时 3D 城市窗口
    python train.py --episodes 400       # 只训练 400 回合
    python train.py --no-render          # 快速训练并保存结果图和模型
"""
import argparse

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "sans-serif"]
matplotlib.rcParams["axes.unicode_minus"] = False

from city_vis import ToyCar, draw_buildings, draw_ground, draw_road_lines
from dqn_agent import DQNAgent
from environment import CarEnv3D


def main():
    parser = argparse.ArgumentParser(description="拟真城市 DQN 城市导航训练")
    parser.add_argument("--episodes", type=int, default=500, help="训练回合数")
    parser.add_argument("--render-every", type=int, default=2,
                        help="每 N 个回合刷新一次画面")
    parser.add_argument("--save", type=str, default="training_result.png",
                        help="最终结果图保存路径")
    parser.add_argument("--model-out", type=str, default="model.pkl",
                        help="训练好的模型保存路径")
    parser.add_argument("--no-render", action="store_true",
                        help="不打开窗口，只训练并保存结果图和模型")
    args = parser.parse_args()

    if args.no_render:
        matplotlib.use("Agg")

    env = CarEnv3D()
    agent = DQNAgent(state_dim=env._state().shape[0], n_actions=env.n_actions)

    # ---------------- 画布 ----------------
    plt.ion()
    fig = plt.figure(figsize=(13, 7))
    ax3d = fig.add_subplot(121, projection="3d")
    ax2d = fig.add_subplot(122)

    L = (env.grid - 1) * env.spacing
    ax3d.set_xlim(-1, L + 1); ax3d.set_ylim(-1, L + 1); ax3d.set_zlim(0, 6)
    ax3d.set_xlabel("X"); ax3d.set_ylabel("Z"); ax3d.set_zlabel("Y")
    ax3d.set_title("拟真城市 · DQN 学习城市导航")

    # 静态城市：地面 + 道路 + 高楼
    draw_ground(ax3d, env.grid, env.spacing)
    draw_road_lines(ax3d, env.grid, env.spacing)
    draw_buildings(ax3d, env.buildings)

    # 目标点（金色）、路径、玩具小车
    goal_pt, = ax3d.plot([], [], [], "yo", markersize=11, label="目标")
    trail, = ax3d.plot([], [], [], "g-", alpha=0.8, lw=2.2, label="小车路径")
    car = ToyCar(ax3d, 0, 0, 0)
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
        traj_world = [env.car_world()]
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
            traj_world.append(env.car_world())

        episode_rewards.append(total_reward)
        avg50 = float(np.mean(episode_rewards[-50:]))

        if not args.no_render and (ep % args.render_every == 0 or ep == 1):
            # 目标 + 路径
            gx, gz = env.goal_world()
            goal_pt.set_data([gx], [gz])
            goal_pt.set_3d_properties([0.6])

            tw = np.array(traj_world)
            trail.set_data(tw[:, 0], tw[:, 1])
            trail.set_3d_properties(np.full(len(tw), 0.4))

            # 小车朝向最后移动方向
            if len(tw) >= 2:
                dx, dz = tw[-1] - tw[-2]
                yaw = np.arctan2(-dz, dx)
            else:
                yaw = 0.0
            cx, cz = env.car_world()
            car.place(cx, cz, yaw)

            # 学习曲线
            xs = list(range(1, len(episode_rewards) + 1))
            line_reward.set_data(xs, episode_rewards)
            if len(episode_rewards) >= 50:
                line_avg.set_data(xs, [np.mean(episode_rewards[max(0, i - 50):i + 1])
                                       for i in range(len(episode_rewards))])
            ax2d.relim(); ax2d.autoscale_view()
            fig.canvas.draw_idle()
            fig.canvas.flush_events()
            plt.pause(0.001)

        if ep % 20 == 0 or ep == 1:
            avg_loss = running_loss / max(loss_count, 1)
            print(f"回合 {ep:>4d} | 本回合奖励 {total_reward:6.1f} | "
                  f"近50回合均值 {avg50:6.1f} | 探索率 {agent.epsilon:.3f} | "
                  f"平均损失 {avg_loss:.4f}")
            running_loss, loss_count = 0.0, 0

    # ---------------- 收尾：补画最后一个回合 + 完整曲线 ----------------
    gx, gz = env.goal_world()
    goal_pt.set_data([gx], [gz]); goal_pt.set_3d_properties([0.6])
    tw = np.array(traj_world)
    trail.set_data(tw[:, 0], tw[:, 1])
    trail.set_3d_properties(np.full(len(tw), 0.4))
    if len(tw) >= 2:
        dx, dz = tw[-1] - tw[-2]
        car.place(*env.car_world(), np.arctan2(-dz, dx))
    else:
        car.place(*env.car_world(), 0.0)

    xs = list(range(1, len(episode_rewards) + 1))
    line_reward.set_data(xs, episode_rewards)
    if len(episode_rewards) >= 50:
        line_avg.set_data(xs, [np.mean(episode_rewards[max(0, i - 50):i + 1])
                               for i in range(len(episode_rewards))])
    ax2d.relim(); ax2d.autoscale_view()
    fig.canvas.draw_idle()

    fig.savefig(args.save, dpi=120, bbox_inches="tight")
    agent.save(args.model_out)
    print(f"\n训练完成！共 {args.episodes} 回合")
    print(f"结果图已保存为 {args.save}")
    print(f"模型已保存为 {args.model_out}（可用 python predict.py 观察城市路径）")

    if args.no_render:
        plt.close(fig)
    else:
        plt.ioff()
        plt.show()


if __name__ == "__main__":
    main()
