"""
城市 3D 可视化辅助 —— 把环境画成拟真的城市。

包含：
- 画地面（道路）
- 画高楼（街区）
- 画玩具小车（车身 + 车顶 + 四个轮子 + 车头方向），可实时移动
"""
import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


def draw_ground(ax, grid, spacing, y=0.0):
    """画浅灰色的城市地面。"""
    L = (grid - 1) * spacing
    xx, zz = np.meshgrid([0.0, L], [0.0, L])
    yy = np.zeros_like(xx) + y
    ax.plot_surface(xx, zz, yy, color="#cfcfcf", alpha=0.55)


def draw_road_lines(ax, grid, spacing, y=0.0):
    """画道路中线（白色网格线），体现道路网。"""
    L = (grid - 1) * spacing
    for i in range(grid):
        p = i * spacing
        ax.plot([p, p], [0, L], [y, y], color="white", lw=1.0, alpha=0.9)
        ax.plot([0, L], [p, p], [y, y], color="white", lw=1.0, alpha=0.9)


def _box_verts(cx, cz, sx, sz, h, y0):
    """返回一个长方体（中心 cx,cz，尺寸 sx×sz，高 h，底 y0）的 6 个面顶点。"""
    x0, x1 = cx - sx / 2, cx + sx / 2
    z0, z1 = cz - sz / 2, cz + sz / 2
    y1 = y0 + h
    return [
        [(x0, y0, z0), (x1, y0, z0), (x1, y0, z1), (x0, y0, z1)],   # 底
        [(x0, y1, z0), (x1, y1, z0), (x1, y1, z1), (x0, y1, z1)],   # 顶
        [(x0, y0, z0), (x1, y0, z0), (x1, y1, z0), (x0, y1, z0)],   # 前
        [(x0, y0, z1), (x1, y0, z1), (x1, y1, z1), (x0, y1, z1)],   # 后
        [(x0, y0, z0), (x0, y0, z1), (x0, y1, z1), (x0, y1, z0)],   # 左
        [(x1, y0, z0), (x1, y0, z1), (x1, y1, z1), (x1, y1, z0)],   # 右
    ]


def draw_buildings(ax, buildings):
    """画出所有高楼（棕色，模拟城市街区）。"""
    faces, colors = [], []
    for cx, cz, sx, sz, h in buildings:
        faces += _box_verts(cx, cz, sx, sz, h, 0.0)
        colors += ["#a8703a"] * 6
    pc = Poly3DCollection(faces, facecolors=colors, edgecolor="#6b4a2a",
                          linewidths=0.2, alpha=0.92)
    ax.add_collection3d(pc)


class ToyCar:
    """玩具小车：车身 + 车顶 + 车头 + 四个轮子，可沿道路移动并转向。"""

    def __init__(self, ax, x, z, yaw, color="dodgerblue"):
        self.ax = ax
        self.color = color
        # 各部件： (局部cx, 局部cz, 长sx, 宽sz, 高h, 底y0, 颜色)
        self.parts = [
            (0.0, 0.0, 0.9, 0.5, 0.30, 0.08, color),          # 车身
            (-0.08, 0.0, 0.42, 0.28, 0.16, 0.38, "#3b82d6"), # 车顶(靠后)
            (0.28, 0.0, 0.18, 0.30, 0.12, 0.08, "#1c4f9c"),  # 车头(前方挡风)
            (-0.32, -0.22, 0.16, 0.10, 0.08, 0.0, "#222"),   # 轮1
            (0.32, -0.22, 0.16, 0.10, 0.08, 0.0, "#222"),    # 轮2
            (-0.32, 0.22, 0.16, 0.10, 0.08, 0.0, "#222"),    # 轮3
            (0.32, 0.22, 0.16, 0.10, 0.08, 0.0, "#222"),     # 轮4
        ]
        self._pc = Poly3DCollection([], edgecolor="k", linewidths=0.2)
        ax.add_collection3d(self._pc)
        self.place(x, z, yaw)

    def _local_faces(self):
        faces, colors = [], []
        for cx, cz, sx, sz, h, y0, c in self.parts:
            faces += _box_verts(cx, cz, sx, sz, h, y0)
            colors += [c] * 6
        return faces, colors

    def place(self, x, z, yaw):
        """把小车放到世界坐标 (x, z)，朝向 yaw（绕 y 轴旋转，0=朝 +x）。"""
        local_faces, colors = self._local_faces()
        world_faces = []
        for face in local_faces:
            wf = []
            for (lx, ly, lz) in face:
                wx = x + lx * np.cos(yaw) + lz * np.sin(yaw)
                wz = z - lx * np.sin(yaw) + lz * np.cos(yaw)
                wf.append((wx, ly, wz))
            world_faces.append(wf)
        self._pc.set_verts(world_faces)
        self._pc.set_facecolors(colors)
