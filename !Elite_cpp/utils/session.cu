import open3d as o3d
import numpy as np
import os
import copy
from typing import List, Optional, Union
from dataclasses import dataclass, field


@dataclass
class Scan:
    pose: np.ndarray
    path: str
    _pcd: Optional[o3d.geometry.PointCloud] = field(default=None, init=False)

    @property
    def pcd(self) -> o3d.geometry.PointCloud:
        if self._pcd is None:
            self._pcd = o3d.io.read_point_cloud(self.path)
            self._pcd.transform(self.pose)
        return self._pcd


class PointCloud:
    def __init__(self, cloud: o3d.geometry.PointCloud):
        self.cloud = cloud

    def downsample(self, voxel_size: float = 0.2) -> 'PointCloud':
        self.cloud = self.cloud.voxel_down_sample(voxel_size)
        return self

    def colorize(self, color: List[float] = [1, 0, 0]) -> 'PointCloud':
        self.cloud.paint_uniform_color(color)
        return self

    def height_filter(self, height: float = 0.5) -> 'PointCloud':
        points = np.asarray(self.cloud.points)
        mask = points[:, 2] > height
        self.cloud = self.cloud.select_by_index(np.where(mask)[0])
        return self

    def visualize(self) -> 'PointCloud':
        mesh = o3d.geometry.TriangleMesh.create_coordinate_frame(size=20, origin=[0, 0, 0])
        o3d.visualization.draw_geometries([self.cloud, mesh])
        return self

    def save(self, path: str) -> 'PointCloud':
        z_values = np.asarray(self.cloud.points)[:, 2]
        intensity = (z_values - np.min(z_values)) / (np.max(z_values) - np.min(z_values) + 1e-8)
        self.cloud.colors = o3d.utility.Vector3dVector(np.c_[intensity, intensity, intensity])
        o3d.io.write_point_cloud(path, self.cloud)
        return self

    def get(self) -> o3d.geometry.PointCloud:
        return copy.deepcopy(self.cloud)


class Session:
    def __init__(self, scans_dir: str, pose_file: str):
        self.scans_dir = scans_dir
        self.pose_file = pose_file
        self.scans = self._load_scans(scans_dir, pose_file)

    def _load_scans(self, scans_dir: str, pose_file: str) -> List[Scan]:
        poses = self._load_poses(pose_file)
        scan_files = sorted([f for f in os.listdir(scans_dir) if f.endswith('.pcd')])

        if len(poses) != len(scan_files):
            raise ValueError("Mismatch between number of poses and scan files.")

        scans = []
        for i, pose in enumerate(poses):
            path = os.path.join(scans_dir, f'{i:06d}.pcd')
            if not os.path.exists(path):
                raise FileNotFoundError(f"Missing scan file: {path}")
            scans.append(Scan(pose, path))
        return scans

    def _load_poses(self, pose_file: str) -> List[np.ndarray]:
        poses = []
        with open(pose_file, 'r') as f:
            for line in f:
                vals = list(map(float, line.strip().split()))
                if len(vals) != 12:
                    raise ValueError(f"Invalid pose line: {line.strip()}")
                pose = np.array(vals).reshape(3, 4)
                pose = np.vstack([pose, [0, 0, 0, 1]])
                poses.append(pose)
        return poses

    def __len__(self):
        return len(self.scans)

    def __getitem__(self, idx: Union[int, slice]) -> PointCloud:
        if isinstance(idx, int):
            scan = self.scans[idx]
            return PointCloud(scan.pcd)
        elif isinstance(idx, slice):
            clouds = [self.scans[i].pcd for i in range(*idx.indices(len(self)))]
            combined = o3d.geometry.PointCloud()
            for pcd in clouds:
                combined += pcd
            return PointCloud(combined)
        else:
            raise TypeError("Index must be int or slice")
    
    def __repr__(self):
        return f"Session with {len(self.scans)} scans."

    def get_pose(self, idx: int) -> np.ndarray:
        return self.scans[idx].pose

    def update_pose(self, idx: int, pose: np.ndarray):
        self.scans[idx].pose = pose

    def save_pose(self, pose_file: str):
        with open(pose_file, 'w') as f:
            for scan in self.scans:
                line = ' '.join(map(str, scan.pose[:3].reshape(-1)))
                f.write(line + '\n')

    def save_pointcloud(self, idx: int, path: str, voxel_size: float = 0.1):
        pc = self[idx].downsample(voxel_size).get()
        z_values = np.asarray(pc.points)[:, 2]
        intensity = (z_values - np.min(z_values)) / (np.max(z_values) - np.min(z_values) + 1e-8)
        pc.colors = o3d.utility.Vector3dVector(np.c_[intensity, intensity, intensity])
        o3d.io.write_point_cloud(path, pc)