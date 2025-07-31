import copy
import pygicp
import numpy as np
import open3d as o3d
from alignment.matcher.base_scan_matcher import BaseScanMatcher


class PyGICPScanMatcher(BaseScanMatcher):
    def __init__(
        self,
        max_correspondence_distance: float = 0.02,
        init_transformation: np.ndarray = None,
        method: str = "GICP",
        downsample_resolution: float = -1.0,
        voxel_resolution: float = 1.0
    ):
        super().__init__(max_correspondence_distance, init_transformation)
        self.method = method
        self.downsample_resolution = downsample_resolution
        self.voxel_resolution = voxel_resolution

    def align(self) -> None:
        if self.source_raw is None or self.target_raw is None:
            raise RuntimeError("Source/target not set.")
        src_arr = np.asarray(self.source_raw.points)
        tgt_arr = np.asarray(self.target_raw.points)
        self.transformation = pygicp.align_points(
            target=tgt_arr,
            source=src_arr,
            method=self.method,
            downsample_resolution=self.downsample_resolution,
            max_correspondence_distance=self.max_correspondence_distance,
            voxel_resolution=self.voxel_resolution,
            k_correspondences=15,
            num_threads=0,
            neighbor_search_method="DIRECT1",
            neighbor_search_radius=1.5,
            initial_guess=self.init_transformation
        )
        pts_h = np.hstack((src_arr, np.ones((src_arr.shape[0],1))))
        transformed = (self.transformation @ pts_h.T).T[:, :3]
        pc = o3d.geometry.PointCloud()
        pc.points = o3d.utility.Vector3dVector(transformed)
        self.transformed = pc