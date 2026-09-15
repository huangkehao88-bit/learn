"""
城市俯视地图可视化 —— 从正上方俯瞰城市，像卫星地图一样清晰。

- 深灰底色 = 道路，白色网格线 = 道路中线
- 棕色方块 = 高楼街区
- 俯视玩具小车（车身 + 车头 + 车窗），可沿道路移动并转向
"""
import numpy as np
import matplotlib.patches as mpatches


def draw_map(ax, grid, spacing):
    """画俯视城市：深灰道路底色 + 白色道路中线。"""
    L = (grid - 1) * spacing
    ax.set_xlim(-1, L + 1)
    ax.set_ylim(-1, L + 1)
    ax.set_aspect("equal")
    ax.set_facecolor("#7d7d7d")                 # 道路底色
    for i in range(grid):                       # 道路中线（白色十字网）
        p = i * spacing
        ax.plot([p, p], [-1, L + 1], color="white", lw=1.6, alpha=0.85)
        ax.plot([-1, L + 1], [p, p], color="white", lw=1.6, alpha=0.85)


def draw_buildings(ax, buildings):
    """在街区内画楼（俯视方块，棕色）。"""
    for cx, cz, sx, sz, h in buildings:
        rect = mpatches.Rectangle((cx - sx / 2, cz - sz / 2), sx, sz,
                                  facecolor="#a8703a", edgecolor="#6b4a2a",
                                  linewidth=0.8, zorder=2)
        ax.add_patch(rect)


class ToyCar2D:
    """俯视玩具小车：车身 + 车头 + 车窗，可整体旋转移动。"""

    # 局部坐标（车头朝 +x）
    LOCAL_BODY = np.array([(-0.40, -0.25), (0.40, -0.25), (0.40, 0.25), (-0.40, 0.25)])
    LOCAL_HOOD = np.array([(0.40, -0.16), (0.58, 0.00), (0.40, 0.16)])   # 车头尖
    LOCAL_WINDOW = np.array([(-0.04, -0.18), (0.20, -0.18), (0.20, 0.18), (-0.04, 0.18)])

    def __init__(self, ax, x, z, yaw):
        specs = [
            (self.LOCAL_BODY, "#1f6fd6", "#0d2f52", 1.2, 5),
            (self.LOCAL_HOOD, "#2b6cb0", "#0d2f52", 1.0, 6),
            (self.LOCAL_WINDOW, "#b8dcff", None, 0.0, 6),
        ]
        self.polys = []
        for local, fc, ec, lw, zorder in specs:
            p = mpatches.Polygon(local.copy(), closed=True, facecolor=fc,
                                 edgecolor=ec, lw=lw, zorder=zorder)
            ax.add_patch(p)
            self.polys.append(p)
        self.place(x, z, yaw)

    def place(self, x, z, yaw):
        """移动到世界坐标 (x, z)，车头朝向 yaw（弧度）。"""
        c, s = np.cos(yaw), np.sin(yaw)
        for poly, local in zip(self.polys,
                               [self.LOCAL_BODY, self.LOCAL_HOOD, self.LOCAL_WINDOW]):
            wx = local[:, 0] * c - local[:, 1] * s + x
            wz = local[:, 0] * s + local[:, 1] * c + z
            poly.set_xy(np.stack([wx, wz], axis=1))
