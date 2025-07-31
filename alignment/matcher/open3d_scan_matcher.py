import copy
import numpy as np
import open3d as o3d
from alignment.matcher.base_scan_matcher import BaseScanMatcher


class Open3DScanMatcher(BaseScanMatcher):
    def __init__(
        self,
        max_correspondence_distance: float = 0.02,
        init_transformation: np.ndarray = None
    ):
        super().__init__(max_correspondence_distance, init_transformation)

    def align(self) -> None:
        if self.source_raw is None or self.target_raw is None:
            raise RuntimeError("Source/target not set." )
        estimator = o3d.pipelines.registration.TransformationEstimationForGeneralizedICP()
        criteria = o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=20)
        reg = o3d.pipelines.registration.registration_generalized_icp(
            self.source_raw, self.target_raw,
            self.max_correspondence_distance,
            self.init_transformation,
            estimator,
            criteria
        )
        self.transformation = reg.transformation
        self.transformed = copy.deepcopy(self.source_raw).transform(self.transformation)