import os
import yaml
import numpy as np
import open3d as o3d
from tqdm import trange
import torch

from utils.session import Session
from utils.session_map import SessionMap
from utils.logger import logger

def knn_cdist_autobatch(scan_pts: torch.Tensor, anchor_pts: torch.Tensor, k: int, max_bytes = 3.5 * 1024 * 1024 * 1024):
    # scan_pts: (N, D)
    # anchor_pts: (A, D)
    # k: top-k
    # max_bytes: 한 번에 연산할 최대 메모리 (float32 기준, default 1GB)


    ### float32로 변환
    if scan_pts.dtype != torch.float32:
        scan_pts = scan_pts.float()
    if anchor_pts.dtype != torch.float32:
        anchor_pts = anchor_pts.float()



    N, D = scan_pts.shape
    A = anchor_pts.shape[0]
    bytes_per_dist = 4  # float32

    total_bytes = N * A * bytes_per_dist

    if total_bytes <= max_bytes:
        dists_all = torch.cdist(scan_pts, anchor_pts)           # (N, A)
        dists_k, inds = torch.topk(dists_all, k, dim=1, largest=False)
        return dists_k, inds

    else:
        # Batch로 나눠서 처리
        print(f"[AutoBatch] Total memory {total_bytes/1e6:.1f}MB > {max_bytes/1e6:.1f}MB → batching")
        batch_size = int(max_bytes // (A * bytes_per_dist))
        dists_k_all, inds_all = [], []

        for i in range(0, N, batch_size):
            end = min(i + batch_size, N)
            dists = torch.cdist(scan_pts[i:end], anchor_pts)       # (B, A)
            d_k, i_k = torch.topk(dists, k, dim=1, largest=False)
            dists_k_all.append(d_k)
            inds_all.append(i_k)

        return torch.cat(dists_k_all, dim=0), torch.cat(inds_all, dim=0)
    
def voxel_downsample_gpu(points: torch.Tensor, voxel_size: float) -> torch.Tensor:
    """
    GPU 상에서 voxel 기반 다운샘플링 수행
    - points: (N, 3) torch.float32 tensor (device=CUDA)
    - voxel_size: float (양수)

    Returns:
    - downsampled_points: (M, 3), 각 voxel의 centroid
    """
    assert points.ndim == 2 and points.shape[1] == 3
    assert points.is_cuda
    assert voxel_size > 0

    # 1. voxel grid 좌표로 quantization
    keys = torch.floor(points / voxel_size).to(torch.int32)  # (N, 3)

    # 2. unique voxel 찾고 inverse 인덱스 추출
    uniq, inv = torch.unique(keys, return_inverse=True, dim=0)  # uniq: (M, 3), inv: (N,)

    # 3. voxel별로 centroid 계산 (index_add + bincount)
    summed = torch.zeros((uniq.shape[0], 3), device=points.device, dtype=points.dtype)
    summed = summed.index_add_(0, inv, points)  # (M, 3)

    counts = torch.bincount(inv, minlength=uniq.shape[0]).unsqueeze(1)  # (M, 1)

    downsampled = summed / counts  # (M, 3)

    return downsampled

class MapRemover:
    def __init__(self, config_path: str):
        with open(config_path, 'r') as f:
            self.params = yaml.safe_load(f)

        p_settings = self.params["settings"]        
        os.makedirs(p_settings["output_dir"], exist_ok=True)
        
        self.std_dev_o = 0.025
        self.std_dev_f = 0.025
        self.alpha = 0.5
        self.beta = 0.1
        self.device = torch.device("cuda")

        self.session_loader: Session = None
        self.session_map: SessionMap = None

    def load(self, new_session: Session = None):
        p_settings = self.params["settings"]
        if new_session is None:
            self.session_loader = Session(p_settings["scans_dir"], p_settings["poses_file"])
        else:
            self.session_loader = new_session
        logger.info(f"Loaded new session")



    def run(self):
        p_settings = self.params["settings"]
        p_dor = self.params["dynamic_object_removal"]
        session_map = self.session_loader[0:len(self.session_loader)].downsample(0.01).get()
        eph_l = np.zeros(len(session_map.points))
        logger.info(f"Initialized session map")



        #앵커포인트
        anchor_points = session_map.voxel_down_sample(p_dor["anchor_voxel_size"])
        anchor_eph_l = np.ones(len(anchor_points.points)) * 0.5

        #gpu로 옮기기
        device = self.device
        anchor_pts = torch.from_numpy(np.asarray(anchor_points.points)).to(device)    # (M,3)
        anchor_eph = torch.from_numpy(anchor_eph_l).to(device)                        # (M,)


        logger.info(f"Updating anchor local ephemerality")

        #메모리 할당 최적화
        eph_sum = torch.zeros_like(anchor_eph)
        eph_cnt = torch.zeros_like(anchor_eph)
        sum_update = torch.zeros_like(anchor_eph)
        count_update = torch.zeros_like(anchor_eph)

        ## 앵커업뎃
        for i in trange(0, len(self.session_loader), p_dor["stride"], desc="Updating \u03B5_l", ncols=100):
            logger.debug(f"Processing scan {i + 1}/{len(self.session_loader)}")
            scan = np.asarray(self.session_loader[i].get().points)
            # 스캔 포인트도 GPU에 올리기
            scan_pts = torch.from_numpy(np.asarray(scan)).to(device)                      # (N,3)
            pose = self.session_loader.get_pose(i)[:3, 3]

            dists_k, inds = knn_cdist_autobatch(scan_pts, anchor_pts, k=p_dor["num_k"])
            alpha = self.alpha
            beta  = self.beta
            std_dev_o = self.std_dev_o
            update_rate = torch.clamp(
                alpha * (1 - torch.exp(-dists_k**2 / std_dev_o**2)) + beta,
                max=alpha
            )  # (N, k)

            eph_prev = anchor_eph[inds]   # (N, k)
            eph_new = eph_prev * update_rate / (
                eph_prev * update_rate + (1 - eph_prev) * (1 - update_rate)
            )

            # scatter_add (sum & count)
            flat_inds = inds.reshape(-1)       # (N*k,)
            flat_vals = eph_new.reshape(-1)   # (N*k,)
            count_add = torch.ones_like(flat_vals)

            eph_sum.zero_()
            eph_cnt.zero_()
            eph_sum.index_add_(0, flat_inds, flat_vals)
            eph_cnt.index_add_(0, flat_inds, torch.ones_like(flat_vals))
            anchor_eph = torch.where(eph_cnt > 0, eph_sum / eph_cnt, anchor_eph)


            pose = torch.as_tensor(self.session_loader.get_pose(i)[:3, 3], device=self.device)
            scan_pts = torch.from_numpy(scan).to(device)

            shifted_scan = scan_pts - pose
            sample_ratios = torch.linspace(p_dor["min_ratio"], p_dor["max_ratio"], p_dor["num_samples"], device=self.device)
            free_space_samples = (pose + shifted_scan.unsqueeze(1) * sample_ratios.unsqueeze(-1)).reshape(-1,3)

            # optional coarse voxel downsample (GPU)
            free_space_samples = voxel_downsample_gpu(free_space_samples, 0.1)


            dists, inds = knn_cdist_autobatch(free_space_samples, anchor_pts, k=p_dor["num_k"])

            update_rate = torch.clamp(
                self.alpha * (1 + torch.exp(-dists**2 / (self.std_dev_f**2))) - self.beta,
                min=self.alpha
            )
            eph_prev = anchor_eph[inds]
            # Bayesian 업데이트 (N',k)
            eph_new = eph_prev * update_rate / (
                eph_prev * update_rate + (1 - eph_prev) * (1 - update_rate)
            )

            # scatter mean 업데이트
            flat_idx = inds.reshape(-1)
            flat_val = eph_new.reshape(-1)

            # 합산
            sum_update.zero_()
            count_update.zero_()

            sum_update.index_add_(0, flat_idx, flat_val)
            count_update.index_add_(0, flat_idx, torch.ones_like(flat_val))
            anchor_eph = torch.where(count_update > 0, sum_update / count_update, anchor_eph)
            
        # 3) Propagate anchor local ephemerality to session map
        map_pts = torch.as_tensor(np.asarray(session_map.points), dtype=torch.float32, device = self.device)
        distances, indices = knn_cdist_autobatch(map_pts, anchor_pts, k=p_dor["num_k"])
        weights = (1 / torch.clamp(distances, min=1e-6)**2)
        weights /= weights.sum(dim=1, keepdim=True)
        eph_l = (weights * anchor_eph[indices]).sum(dim=1).clamp_(0,1)  

        # 4) Remove dynamic objects to create cleaned session map
        mask_static = eph_l <= p_dor["dynamic_threshold"]
        static_np   = map_pts[mask_static].cpu().numpy()
        static_eph_l  = eph_l[mask_static].cpu().numpy()
        static_points   = o3d.geometry.PointCloud(
                        o3d.utility.Vector3dVector(static_np))
        dynamic_points      = o3d.geometry.PointCloud(
                        o3d.utility.Vector3dVector(map_pts[~mask_static].cpu().numpy()))
        static_points.paint_uniform_color([.5,.5,.5]); dynamic_points.paint_uniform_color([1,0,0])
                  
        if p_dor["save_static_dynamic_map"]:
            o3d.io.write_point_cloud(os.path.join(p_settings["output_dir"], "static_points.pcd"), static_points)
            o3d.io.write_point_cloud(os.path.join(p_settings["output_dir"], "dynamic_points.pcd"), dynamic_points)
        if p_dor["viz_static_dynamic_map"]:
            total_points = static_points + dynamic_points
            o3d.visualization.draw_geometries([total_points])

        cleaned_session_map = SessionMap(np.asarray(static_points.points), static_eph_l)
        self.session_map = cleaned_session_map

        if p_dor["save_cleaned_session_map"]:
            cleaned_session_map.save(p_settings["output_dir"], is_global=False)
        if p_dor["viz_cleaned_session_map"]:
            cleaned_session_map.visualize()

        return cleaned_session_map

    def get(self):
        return self.session_map


if __name__ == "__main__":
    config = "../config/sample.yaml"
    remover = MapRemover(config)
    remover.load()
    remover.run()
    cleaned_session_map = remover.get()
