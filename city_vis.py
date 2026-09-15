"""
拟真城市 3D 斜俯视可视化 —— 从斜上方俯瞰立体城市。

- 深灰底面 = 道路，白色网格线 = 道路中线
- 立体棕色长方体 = 高楼街区
- 3D 玩具小车（车身 + 车顶 + 车头 + 四轮），可沿道路行驶并转向
"""
import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


def set_isometric_view(ax, elev=62, azim=-45):
    """设置俯视透视视角：近处楼低、远处楼高，有纵深，又能看清路径。"""
    ax.view_init(elev=elev, azim=azim)
    try:
        ax.set_proj_type("persp")      # 透视投影，产生近大远小（近低远高）的纵深
    except Exception:
        pass


def draw_ground(ax, node_xy):
    """画底面（道路）。node_xy 为交叉口世界坐标 (grid, grid, 2)。"""
    xs = node_xy[:, :, 0]
    zs = node_xy[:, :, 1]
    xx, zz = np.meshgrid([xs.min() - 0.7, xs.max() + 0.7],
                         [zs.min() - 0.7, zs.max() + 0.7])
    ax.plot_surface(xx, zz, np.zeros_like(xx), color="#7d7d7d", alpha=0.9)


def draw_roads(ax, road_edges):
    """画不规则道路（相邻交叉口之间的街道，白色，贴在地面）。"""
    for (x1, z1), (x2, z2) in road_edges:
        ax.plot([x1, x2], [z1, z2], [0, 0], color="white", lw=1.3, alpha=0.85)


def _box_verts(cx, cz, sx, sz, h, y0):
    """长方体 6 个面。坐标系：x 横、y 纵为地面（水平面），高度 h 为 z 轴。"""
    x0, x1 = cx - sx / 2, cx + sx / 2     # matplotlib x（横）
    z0, z1 = cz - sz / 2, cz + sz / 2     # matplotlib y（纵，世界 z）
    zbot, ztop = y0, y0 + h               # matplotlib z（高度）
    return [
        [(x0, z0, zbot), (x1, z0, zbot), (x1, z1, zbot), (x0, z1, zbot)],
        [(x0, z0, ztop), (x1, z0, ztop), (x1, z1, ztop), (x0, z1, ztop)],
        [(x0, z0, zbot), (x1, z0, zbot), (x1, z0, ztop), (x0, z0, ztop)],
        [(x0, z1, zbot), (x1, z1, zbot), (x1, z1, ztop), (x0, z1, ztop)],
        [(x0, z0, zbot), (x0, z0, ztop), (x0, z1, ztop), (x0, z1, zbot)],
        [(x1, z0, zbot), (x1, z0, ztop), (x1, z1, ztop), (x1, z1, zbot)],
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
        """把小车放到世界坐标 (x, z)，车头朝向 yaw（绕高度 z 轴旋转）。"""
        c, s = np.cos(yaw), np.sin(yaw)
        faces, colors = [], []
        for cx, cz, sx, sz, h, y0, col in self.parts:
            for face in _box_verts(cx, cz, sx, sz, h, y0):
                wf = []
                for (lx, ly, lz) in face:
                    # lx=横, ly=纵(世界z), lz=高度；旋转作用于水平面
                    wx = x + lx * c + ly * s
                    wy = z - lx * s + ly * c
                    wf.append((wx, wy, lz))
                faces.append(wf)
                colors.append(col)
        self._pc.set_verts(faces)
        self._pc.set_facecolors(colors)
