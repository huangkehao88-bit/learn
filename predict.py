"""
推理演示 —— 3D 俯视透视城市自动驾驶，实时摄像机跟随小车。

演示内容：
- 小车沿不规则道路自动寻路到目标（绿色路径）
- 遇到红灯会停下等待（红绿灯红/绿切换）
- 前方有其他车时让行等待，安全会车
- 固定全景视角展示整个城市

用法：
    python predict.py
    python predict.py --model my.pkl
"""
import argparse

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "sans-serif"]
matplotlib.rcParams["axes.unicode_minus"] = False

from city_vis import (ToyCar, draw_buildings, draw_ground, draw_roads,
                      draw_signals, set_isometric_view, update_signals)
from dqn_agent import DQNAgent
from environment import CarEnv3D

NPC_COLORS = ["crimson", "darkorange", "rebeccapurple"]


def main():
    parser = argparse.ArgumentParser(description="拟真城市自动驾驶推理演示")
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

    # ---------------- 画布（3D 俯视透视）----------------
    plt.ion()
    fig = plt.figure(figsize=(12.5, 6.5))
    ax = fig.add_subplot(121, projection="3d")
    ax2d = fig.add_subplot(122)

    xmin, xmax, zmin, zmax = env.world_bounds
    ax.set_xlim(xmin, xmax); ax.set_ylim(zmin, zmax); ax.set_zlim(0, 4)
    ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")
    ax.set_title("3D 俯视透视城市 · 自动驾驶（红灯/会车）")
    draw_ground(ax, env.node_xy)
    draw_roads(ax, env.road_edges)
    draw_buildings(ax, env.buildings)
    signal_artists = draw_signals(ax, env)
    npc_cars = [ToyCar(ax, *env._world(npc["pos"]), 0, color=NPC_COLORS[k % 3])
                for k, npc in enumerate(env.npc_cars)]
    set_isometric_view(ax)

    goal_pt, = ax.plot([], [], [], "o", color="gold", ms=11, mec="k",
                       mew=1.2, zorder=4, label="目标")
    trail, = ax.plot([], [], [], "-", color="#2ecc40", lw=2.6,
                     zorder=3, label="小车路径")
    car = ToyCar(ax, 0, 0, 0)
    ax.legend(loc="upper left")

    ax2d.set_title("每个演示回合 · 小车到目标的距离")
    ax2d.set_xlabel("步数"); ax2d.set_ylabel("到目标距离")
    ax2d.grid(True, alpha=0.4)

    # ---------------- 推理循环（观察自动驾驶）----------------
    all_distances, path_cells, episode_goals = [], [], []
    waits = []                                  # 记录每回合的等灯/让行次数

    for ep in range(1, args.episodes + 1):
        state = env.reset()
        episode_goals.append(env.goal)
        path = [env.car]
        traj_world = [env.car_world()]
        dists = [env._dist(env.car, env.goal)]
        reached = False
        done = False
        wait_count = 0
        prev_car = env.car

        while not done:
            action = agent.act(state, training=False)
            state, reward, done, info = env.step(action)
            if env.car == prev_car:             # 小车没动 = 在等灯/让行
                wait_count += 1
            prev_car = env.car
            path.append(env.car)
            traj_world.append(env.car_world())
            dists.append(info["dist"])
            if env.car == env.goal:
                reached = True

            if not args.no_render:
                cx, cz = env.car_world()
                gx, gz = env.goal_world()
                goal_pt.set_data([gx], [gz]); goal_pt.set_3d_properties([0.5])
                tw = np.array(traj_world)
                trail.set_data(tw[:, 0], tw[:, 1])
                trail.set_3d_properties(np.full(len(tw), 0.12))
                if len(tw) >= 2:
                    dx, dz = tw[-1] - tw[-2]
                    car.place(cx, cz, np.arctan2(-dz, dx))
                update_signals(signal_artists, env)
                for k, npc in enumerate(env.npc_cars):
                    route = npc["route"]; idx = npc["idx"]
                    prev = route[(idx - 1) % len(route)]
                    cur = npc["pos"]
                    npc_cars[k].place(*env._world(cur),
                                      np.arctan2(-(cur[1] - prev[1]), cur[0] - prev[0]))
                fig.canvas.draw_idle()
                fig.canvas.flush_events()
                plt.pause(0.3)

        all_distances.append(dists)
        path_cells.append(path)
        waits.append(wait_count)
        steps = " → ".join(f"{c[0]},{c[1]}" for c in path)
        print(f"回合 {ep}: {'到达' if reached else '未达'} | 用时 {len(path)-1} 步 | "
              f"等灯/让行 {wait_count} 次 | 路径 {steps}")

    # ---------------- 收尾：拉回全景 + 距离曲线 ----------------
    ax.set_xlim(xmin, xmax); ax.set_ylim(zmin, zmax)
    gx, gz = env.goal_world()
    goal_pt.set_data([gx], [gz]); goal_pt.set_3d_properties([0.5])
    tw = np.array(traj_world)
    trail.set_data(tw[:, 0], tw[:, 1])
    trail.set_3d_properties(np.full(len(tw), 0.12))
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
