"""
推理演示 —— 俯视城市地图，观察玩具小车自动找到的最优路径。

小车从起点沿道路行驶，自动绕开高楼街区到达目标（金色点）。
左侧俯视城市清晰展示小车走的路径（绿色），右侧显示到目标的距离下降曲线。

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

from city_vis import ToyCar2D, draw_buildings, draw_map
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

    # ---------------- 画布（俯视）----------------
    plt.ion()
    fig = plt.figure(figsize=(12, 6))
    ax = fig.add_subplot(121)
    ax2d = fig.add_subplot(122)

    ax.set_xlabel("X"); ax.set_ylabel("Z")
    ax.set_title("俯视城市 · 小车自动寻路到目标")
    draw_map(ax, env.grid, env.spacing)
    draw_buildings(ax, env.buildings)
    goal_pt, = ax.plot([], [], "o", color="gold", ms=12, mec="k",
                       mew=1.2, zorder=4, label="目标")
    trail, = ax.plot([], [], "-", color="#2ecc40", lw=2.4, zorder=3, label="小车路径")
    car = ToyCar2D(ax, 0, 0, 0)
    ax.legend(loc="upper left")

    ax2d.set_title("每个演示回合 · 小车到目标的距离")
    ax2d.set_xlabel("步数"); ax2d.set_ylabel("到目标距离")
    ax2d.grid(True, alpha=0.4)

    # ---------------- 推理循环（观察路径）----------------
    all_distances, path_cells, episode_goals = [], [], []

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
                goal_pt.set_data([gx], [gz])
                tw = np.array(traj_world)
                trail.set_data(tw[:, 0], tw[:, 1])
                if len(tw) >= 2:
                    dx, dz = tw[-1] - tw[-2]
                    car.place(*env.car_world(), np.arctan2(dz, dx))
                fig.canvas.draw_idle()
                fig.canvas.flush_events()
                plt.pause(0.35)

        all_distances.append(dists)
        path_cells.append(path)
        steps = " → ".join(f"{c[0]},{c[1]}" for c in path)
        print(f"回合 {ep}: {'到达目标' if reached else '未到达'} | "
              f"用时 {len(path) - 1} 步 | 路径 {steps}")

    # ---------------- 收尾：画最后一个回合路径 + 距离曲线 ----------------
    gx, gz = env.goal_world()
    goal_pt.set_data([gx], [gz])
    tw = np.array(traj_world)
    trail.set_data(tw[:, 0], tw[:, 1])
    if len(tw) >= 2:
        dx, dz = tw[-1] - tw[-2]
        car.place(*env.car_world(), np.arctan2(dz, dx))

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
