import numpy as np
import open3d as o3d
from copy import deepcopy
from utils.logger import logger

def preprocess_point_cloud(pcd, voxel_size):
    pcd_down = pcd.voxel_down_sample(voxel_size)
    pcd_down.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 2.0, max_nn=30)
    )
    pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd_down,
        o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 5.0, max_nn=100),
    )
    return pcd_down, pcd_fpfh

def visualize_registration(src, dst, transformation=np.eye(4)):

    src_orig = deepcopy(src)
    src_orig.paint_uniform_color([1, 0, 0]) # Red color for the original source

    src_trans = deepcopy(src)
    src_trans.transform(transformation)
    src_trans.paint_uniform_color([0, 0, 1]) # Blue color for the transformed source

    dst_clone = deepcopy(dst)
    dst_clone.paint_uniform_color([0, 0, 0]) # Black color for the target

    vis = o3d.visualization.Visualizer()
    vis.create_window()
    vis.add_geometry(src_orig)
    vis.add_geometry(src_trans)
    vis.add_geometry(dst_clone)
    vis.run()


def register_with_fpfh_ransac(src, dst, voxel_size=0.05, distance_multiplier=1.5, 
                           max_iterations=100000, confidence=0.999, mutual_filter=False):
    distance_threshold = distance_multiplier * voxel_size
    # o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Debug)

    logger.info("Downsampling inputs")
    src_down, src_fpfh = preprocess_point_cloud(src, voxel_size)
    dst_down, dst_fpfh = preprocess_point_cloud(dst, voxel_size)

    logger.info("Running RANSAC from features")
    result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        src_down,
        dst_down,
        src_fpfh,
        dst_fpfh,
        mutual_filter=mutual_filter,
        max_correspondence_distance=distance_threshold,
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        ransac_n=3,
        checkers=[
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(distance_threshold),
        ],
        criteria=o3d.pipelines.registration.RANSACConvergenceCriteria(max_iterations, confidence),
    )

    # visualize_registration(src, dst, result.transformation)

    return result