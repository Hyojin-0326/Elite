import copy
import numpy as np
import open3d as o3d
from typing import Optional
from abc import ABC, abstractmethod


class BaseScanMatcher(ABC):
    def __init__(self, max_correspondence_distance: float = 0.02, init_transformation: np.ndarray = None):
        self.max_correspondence_distance = max_correspondence_distance
        self.init_transformation = init_transformation if init_transformation is not None else np.eye(4)
        self.source_raw: Optional[o3d.geometry.PointCloud] = None
        self.target_raw: Optional[o3d.geometry.PointCloud] = None
        self.transformation: Optional[np.ndarray] = None
        self.transformed: Optional[o3d.geometry.PointCloud] = None

    def set_input_src(self, src: o3d.geometry.PointCloud) -> None:
        self.source_raw = copy.deepcopy(src)

    def set_input_tgt(self, tgt: o3d.geometry.PointCloud) -> None:
        self.target_raw = copy.deepcopy(tgt)

    @abstractmethod
    def align(self) -> None:
        """Perform alignment; must set self.transformation and self.transformed."""
        pass

    def get_final_transformation(self) -> np.ndarray:
        if self.transformation is None:
            raise RuntimeError("No transformation available; call align() first.")
        return self.transformation

    def get_transformed_src(self) -> o3d.geometry.PointCloud:
        if self.transformed is None:
            raise RuntimeError("No transformed source; call align() first.")
        return self.transformed

    def get_registration_result(self) -> tuple:
        if self.transformation is None:
            raise RuntimeError("No result available; call align() first.")
        eval = o3d.pipelines.registration.evaluate_registration(
            self.source_raw, self.target_raw,
            self.max_correspondence_distance,
            self.transformation
        )
        return eval.fitness, eval.inlier_rmse