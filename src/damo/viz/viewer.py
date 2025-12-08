import time
from typing import Sequence, List, Tuple
import numpy as np
import open3d as o3d

from ..utils.ensure_types import ensure_numpy

class Viewer:
    def __init__(
        self,
        width: int = 960,
        height: int = 720,
        bg_color: Sequence[float] = (1, 1, 1),
        smooth_normals: bool = False,
        show_wireframe: bool = True,
        show_back_face: bool = False,
        show_axis: bool = True,
        axis_size: float = 1.0,
    ) -> None:
        self.width = width
        self.height = height
        self.bg_color = tuple(bg_color)
        self.smooth_normals = smooth_normals
        self.show_wireframe = show_wireframe
        self.show_back_face = show_back_face
        self.show_axis = show_axis
        self.axis_size = axis_size

        self._static_geoms: List[o3d.geometry.Geometry] = []
        self._meshes: List[Tuple[o3d.geometry.TriangleMesh, np.ndarray]] = []
        self._points: List[Tuple[o3d.geometry.PointCloud, np.ndarray]] = []

    def clear(self) -> None:
        self._static_geoms.clear()
        self._meshes.clear()
        self._points.clear()

    def add_mesh(
        self,
        vertices: np.ndarray,
        faces: np.ndarray,
        color: Sequence[float] | None = None,
    ) -> o3d.geometry.TriangleMesh:
        v = ensure_numpy(vertices)
        f = self._ensure_faces(faces)

        if v.ndim == 2:
            v = v[None, :, :]

        mesh = o3d.geometry.TriangleMesh()
        mesh.vertices = o3d.utility.Vector3dVector(v[0])
        mesh.triangles = o3d.utility.Vector3iVector(f)

        if self.smooth_normals:
            mesh.compute_vertex_normals()

        if color is not None:
            mesh.paint_uniform_color(np.asarray(color, dtype=np.float64))

        self._meshes.append((mesh, v))
        return mesh

    def add_points(
        self,
        points: np.ndarray,
        color: np.ndarray | None = None,
    ) -> o3d.geometry.PointCloud:
        pts = ensure_numpy(points)
        pcd = o3d.geometry.PointCloud()

        if pts.ndim == 2:
            pts = pts[None, :, :]

        pcd.points = o3d.utility.Vector3dVector(pts[0])

        if color is not None:
            # uniform color
            col = np.asarray(color, dtype=np.float64)

            if col.ndim == 1:
                col = np.tile(col[None, :], (pts.shape[1], 1))
            else:
                assert col.shape[0] == pts.shape[1]

            pcd.colors = o3d.utility.Vector3dVector(col)

        self._points.append((pcd, pts))
        return pcd

    def run(self, fps=60.0, repeat=True) -> None:
        vis = o3d.visualization.Visualizer()
        vis.create_window(
            window_name="Viewer",
            width=self.width,
            height=self.height,
        )

        try:
            opt = vis.get_render_option()
            opt.background_color = np.asarray(self.bg_color, dtype=np.float32)
            opt.mesh_show_wireframe = bool(self.show_wireframe)
            opt.mesh_show_back_face = bool(self.show_back_face)

            geoms: List[o3d.geometry.Geometry] = list(self._static_geoms)
            for mesh, _ in self._meshes:
                geoms.append(mesh)
            for pcd, _ in self._points:
                geoms.append(pcd)
            if self.show_axis:
                geoms.append(self._make_axis())

            for i, g in enumerate(geoms):
                vis.add_geometry(g, reset_bounding_box=(i == 0))

            mesh_T = [seq.shape[0] for _, seq in self._meshes]
            points_T = [seq.shape[0] for _, seq in self._points]
            max_T = max(mesh_T + points_T)

            t = 0
            frame_dt = 1.0 / max(1e-6, float(fps))
            running = True
            while running:
                start = time.perf_counter()

                if t >= max_T:
                    if not repeat:
                        break
                    t = 0

                for mesh, v_seq in self._meshes:
                    idx = min(t, v_seq.shape[0] - 1)
                    mesh.vertices = o3d.utility.Vector3dVector(v_seq[idx])
                    if self.smooth_normals:
                        mesh.compute_vertex_normals()
                    vis.update_geometry(mesh)

                for pcd, p_seq in self._points:
                    idx = min(t, p_seq.shape[0] - 1)
                    pcd.points = o3d.utility.Vector3dVector(p_seq[idx])
                    vis.update_geometry(pcd)

                if not vis.poll_events():
                    running = False
                    break
                vis.update_renderer()

                elapsed = time.perf_counter() - start
                if frame_dt > elapsed:
                    time.sleep(frame_dt - elapsed)

                t += 1

        finally:
            vis.destroy_window()

    def vis_frame(self, vertices: np.ndarray, faces: np.ndarray) -> None:
        self.clear()
        self.add_mesh(vertices, faces)
        self.run()

    def _make_axis(self) -> o3d.geometry.TriangleMesh:
        return o3d.geometry.TriangleMesh.create_coordinate_frame(
            size=self.axis_size, origin=[0, 0, 0]
        )

    @staticmethod
    def _ensure_faces(faces: np.ndarray) -> np.ndarray:
        faces = np.asarray(faces, dtype=np.int32)
        if faces.ndim != 2 or faces.shape[1] != 3:
            raise ValueError(f"faces must be (F,3), got {faces.shape}")
        if faces.min() < 0:
            raise ValueError("faces must be 0-based indices (found negative).")
        return faces
