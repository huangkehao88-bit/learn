"""
拟真城市 3D 斜俯视可视化 —— 从斜上方俯瞰立体城市。

- 深灰底面 = 道路，白色网格线 = 道路中线
- 立体棕色长方体 = 高楼街区
- 3D 玩具小车（车身 + 车顶 + 车头 + 四轮），可沿道路行驶并转向
"""
import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


def set_isometric_view(ax, elev=40, azim=-58):
    """设置斜俯视视角（等距感，能看到楼高又能看清道路）。"""
    ax.view_init(elev=elev, azim=azim)


def draw_ground(ax, street_x, street_z):
    """画底面（道路）。street_x / street_z 为街道位置数组。"""
    xx, zz = np.meshgrid([0.0, street_x[-1]], [0.0, street_z[-1]])
    yy = np.zeros_like(xx)
    ax.plot_surface(xx, zz, yy, color="#7d7d7d", alpha=0.9)


def draw_road_lines(ax, street_x, street_z):
    """画道路中线（白色，沿不规则街道，贴在地面）。"""
    xmax, zmax = street_x[-1], street_z[-1]
    for p in street_x:
        ax.plot([p, p], [0, zmax], [0, 0], color="white", lw=1.4, alpha=0.85)
    for p in street_z:
        ax.plot([0, xmax], [p, p], [0, 0], color="white", lw=1.4, alpha=0.85)


def _box_verts(cx, cz, sx, sz, h, y0):
    """返回长方体（中心 cx,cz，尺寸 sx×sz，高 h，底 y0）的 6 个面。"""
    x0, x1 = cx - sx / 2, cx + sx / 2
    z0, z1 = cz - sz / 2, cz + sz / 2
    y1 = y0 + h
    return [
        [(x0, y0, z0), (x1, y0, z0), (x1, y0, z1), (x0, y0, z1)],
        [(x0, y1, z0), (x1, y1, z0), (x1, y1, z1), (x0, y1, z1)],
        [(x0, y0, z0), (x1, y0, z0), (x1, y1, z0), (x0, y1, z0)],
        [(x0, y0, z1), (x1, y0, z1), (x1, y1, z1), (x0, y1, z1)],
        [(x0, y0, z0), (x0, y0, z1), (x0, y1, z1), (x0, y1, z0)],
        [(x1, y0, z0), (x1, y0, z1), (x1, y1, z1), (x1, y1, z0)],
    ]


def draw_buildings(ax, buildings):
    """画出所有立体高楼（棕色长方体，半透明以减少遮挡）。"""
    faces, colors = [], []
    for cx, cz, sx, sz, h in buildings:
        faces += _box_verts(cx, cz, sx, sz, h, 0.0)
        colors += ["#b0713c"] * 6
    pc = Poly3DCollection(faces, facecolors=colors, edgecolor="#6b4a2a",
                          linewidths=0.2, alpha=0.82)
    ax.add_collection3d(pc)


class ToyCar:
    """3D 玩具小车：车身 + 车顶 + 车头 + 四个轮子，可沿道路行驶并转向。"""

    def __init__(self, ax, x, z, yaw, color="dodgerblue"):
        # 各部件： (局部cx, 局部cz, 长sx, 宽sz, 高h, 底y0, 颜色)
        self.parts = [
            (0.0, 0.0, 0.9, 0.5, 0.30, 0.08, color),          # 车身
            (-0.08, 0.0, 0.42, 0.28, 0.16, 0.38, "#3b82d6"), # 车顶(靠后)
            (0.28, 0.0, 0.18, 0.30, 0.12, 0.08, "#1c4f9c"),  # 车头
            (-0.32, -0.22, 0.16, 0.10, 0.08, 0.0, "#222"),   # 轮1
            (0.32, -0.22, 0.16, 0.10, 0.08, 0.0, "#222"),    # 轮2
            (-0.32, 0.22, 0.16, 0.10, 0.08, 0.0, "#222"),    # 轮3
            (0.32, 0.22, 0.16, 0.10, 0.08, 0.0, "#222"),     # 轮4
        ]
        self._pc = Poly3DCollection([], edgecolor="k", linewidths=0.2)
        ax.add_collection3d(self._pc)
        self.place(x, z, yaw)

    def place(self, x, z, yaw):
        """把小车放到世界坐标 (x, z)，车头朝向 yaw（弧度，绕 y 轴旋转）。"""
        c, s = np.cos(yaw), np.sin(yaw)
        faces, colors = [], []
        for cx, cz, sx, sz, h, y0, col in self.parts:
            for face in _box_verts(cx, cz, sx, sz, h, y0):
                wf = []
                for (lx, ly, lz) in face:
                    wx = x + lx * c + lz * s
                    wz = z - lx * s + lz * c
                    wf.append((wx, ly, wz))
                faces.append(wf)
                colors.append(col)
        self._pc.set_verts(faces)
        self._pc.set_facecolors(colors)
