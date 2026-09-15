"""
推理演示 —— 加载训练好的模型，观察玩具小车在拟真城市里自动找到的最优路径。

小车从起点沿道路行驶，自动绕开高楼街区，到达目标（金色点）。
左侧 3D 城市完整展示小车走的路径，右侧显示到目标的距离下降曲线。

用法：
    python predict.py                  # 加载 model.pkl 观察城市路径
    python predict.py --model my.pkl   # 指定模型
    python predict.py --no-render      # 不弹窗，只跑结果并保存图
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
    parser = argparse.ArgumentParser(description="拟真城市导航推理演示")
    parser.add_argument("--model", type=str, default="model.pkl")
    parser.add_argument("--episodes", type=int, default=3, help="演示回合数")
    parser.add_argument("--save", type=str, default="predict_result.png")
    parser.add_argument("--no-render", action="store_true")
    args = parser.parse_args()

    if args.no_render:
        matplotlib.use("Agg")

    env = CarEnv3D()
    agent = DQNAgent.load(args.model)
    agent.epsilon = 0.0
    print(f"已加载模型 {args.model}（学习步数 {agent.learn_steps}）")

    # ---------------- 画布 ----------------
    plt.ion()
    fig = plt.figure(figsize=(13, 7))
    ax3d = fig.add_subplot(121, projection="3d")
    ax2d = fig.add_subplot(122)

    L = (env.grid - 1) * env.spacing
    ax3d.set_xlim(-1, L + 1); ax3d.set_ylim(-1, L + 1); ax3d.set_zlim(0, 6)
    ax3d.set_xlabel("X"); ax3d.set_ylabel("Z"); ax3d.set_zlabel("Y")
    ax3d.set_title("拟真城市 · 小车自动寻路到目标")

    draw_ground(ax3d, env.grid, env.spacing)
    draw_road_lines(ax3d, env.grid, env.spacing)
    draw_buildings(ax3d, env.buildings)

    goal_pt, = ax3d.plot([], [], [], "yo", markersize=11, label="目标")
    trail, = ax3d.plot([], [], [], "g-", alpha=0.8, lw=2.2, label="小车路径")
    car = ToyCar(ax3d, 0, 0, 0)
    ax3d.legend(loc="upper left")

    ax2d.set_title("每个演示回合 · 小车到目标的距离")
    ax2d.set_xlabel("步数"); ax2d.set_ylabel("到目标距离")
    ax2d.grid(True, alpha=0.4)

    # ---------------- 推理循环（观察路径）----------------
    all_distances, last_dists, path_cells, episode_goals = [], [], [], []

    for ep in range(1, args.episodes + 1):
        state = env.reset()
        episode_goals.append(env.goal)
        path = [env.car]                      # 路径格点序列
        traj_world = [env.car_world()]
        dists = [env._dist(env.car, env.goal)]
        reached = False
        done = False
        while not done:
            action = agent.act(state, training=False)
            state, _, done, info = env.step(action)
            path.append(env.car)
            traj_world.append(env.car_world())
            dists.append(info["dist"])
            if env.car == env.goal:
                reached = True

            if not args.no_render:
                gx, gz = env.goal_world()
                goal_pt.set_data([gx], [gz]); goal_pt.set_3d_properties([0.6])
                tw = np.array(traj_world)
                trail.set_data(tw[:, 0], tw[:, 1])
                trail.set_3d_properties(np.full(len(tw), 0.4))
                if len(tw) >= 2:
                    dx, dz = tw[-1] - tw[-2]
                    car.place(*env.car_world(), np.arctan2(-dz, dx))
                fig.canvas.draw_idle()
                fig.canvas.flush_events()
                plt.pause(0.3)

        all_distances.append(dists)
        last_dists.append(dists[-1])
        path_cells.append(path)
        steps = [f"{c[0]},{c[1]}" for c in path]
        print(f"回合 {ep}: {'到达目标' if reached else '未到达'} | "
              f"用时 {len(path)-1} 步 | 路径 {steps}")

    # ---------------- 收尾：画路径 + 距离曲线 ----------------
    gx, gz = env.goal_world()
    goal_pt.set_data([gx], [gz]); goal_pt.set_3d_properties([0.6])
    tw = np.array(traj_world)
    trail.set_data(tw[:, 0], tw[:, 1]); trail.set_3d_properties(np.full(len(tw), 0.4))
    if len(tw) >= 2:
        dx, dz = tw[-1] - tw[-2]
        car.place(*env.car_world(), np.arctan2(-dz, dx))

    for dists in all_distances:
        ax2d.plot(list(range(len(dists))), dists, alpha=0.7, lw=1.5)
    ax2d.relim(); ax2d.autoscale_view()
    fig.canvas.draw_idle()
    fig.savefig(args.save, dpi=120, bbox_inches="tight")

    succ = sum(1 for p, g in zip(path_cells, episode_goals) if p[-1] == g)
    avg_steps = np.mean([len(p) - 1 for p in path_cells])
    print(f"\n演示完成：{args.episodes} 回合中 {succ} 次到达目标"
          f"（成功率 {succ / args.episodes:.0%}），平均用时 {avg_steps:.1f} 步")
    print(f"结果图已保存为 {args.save}")

    if args.no_render:
        plt.close(fig)
    else:
        plt.ioff()
        plt.show()


if __name__ == "__main__":
    main()
