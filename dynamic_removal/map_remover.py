import os
import yaml
import numpy as np
import open3d as o3d
from tqdm import trange
from scipy.spatial import KDTree

from utils.session import Session
from utils.session_map import SessionMap
from utils.logger import logger


class MapRemover:
    def __init__(
        self, 
        config_path: str
    ):
        # Load parameters
        with open(config_path, 'r') as f:
            self.params = yaml.safe_load(f)

        p_settings = self.params["settings"]        
        os.makedirs(p_settings["output_dir"], exist_ok=True)
        
        self.std_dev_o = 0.025
        self.std_dev_f = 0.025
        self.alpha = 0.5
        self.beta = 0.1

        self.session_loader : Session = None
        self.session_map : SessionMap = None


    def load(self, new_session : Session = None):
        p_settings = self.params["settings"]

        if new_session is None:
            self.session_loader = Session(p_settings["scans_dir"], p_settings["poses_file"])
        else:
            self.session_loader = new_session
        
        logger.info(f"Loaded new session")


    def run(self):
        p_settings = self.params["settings"]
        p_dor = self.params["dynamic_object_removal"]

        # 1) Aggregate scans to create session map
        session_map = self.session_loader[0:len(self.session_loader)].downsample(0.01).get() 
        #(M, 3)의 clouds 행렬 로드하고 1cm 복셀그리드 centroid, M은 현재 세션의 포인트 갯수
        eph_l = np.zeros(len(session_map.points)) #M개만큼 0으로 만들어놓음, 현재 세션의 모든 포인트의 eph 맵
        logger.info(f"Initialized session map")

        # 2) Select anchor points for local ephemerality update < 앵커에 대해서만 한 번 업뎃함
        anchor_points = session_map.voxel_down_sample(p_dor["anchor_voxel_size"]) 
        #한 번 더 다운샘플링, 얘네가 주변 점 ephemerality 업데이트의 앵커역할 함. A개(A<M)로 전체 포인트 수 줄어듦
        anchor_eph_l = np.ones(len(anchor_points.points)) * 0.5 
        # initial value, 앵커 eph A개 모두 0.5로 둠
        anchor_kdtree = KDTree(np.asarray(anchor_points.points))

        logger.info(f"Updating anchor local ephemerality")
        for i in trange(0, len(self.session_loader), p_dor["stride"], desc="Updating \u03B5_l", ncols=100):
        # session_loader: 1개의 세션을 로드함. length: 그 세션의 스캔수
            logger.debug(f"Processing scan {i + 1}/{len(self.session_loader)}")
            scan = np.asarray(self.session_loader[i].get().points)
            # 세션의 i번째 스캔을 꺼내고 scan행렬: (N_i, 3) 을 만듦. (N_i는 i번째 세션의 pcd 수)
            pose = self.session_loader.get_pose(i)[:3, 3]
            #pose:T행렬, i번째 스캔의 T행렬만 가져옴 (3,)
            
            # occupied space update
            dists, inds = anchor_kdtree.query(scan, k=p_dor["num_k"]) #scan 포인트 각각에 가까운 k 앵커를 찾음
            #scan: i번째 스캔의 포인트들(N_i, 3)
            #dists[m][n]: m번째 포인트와 n번째 앵커점 사이의 거리
            #ids[m][n]: m번째 포인트와 n번째 앵커점 연결의 인덱스(바이너리 그래프) << 그래프 연산 처리? ephe를 그래프 업데이트로 계산할수도 ...

            for j in range(len(dists)):
                #N_i개의 포인트를 모두 돎(j번째 포인트)
                dist = dists[j] # j번째 포인트와 가까운 k개의 앵커점 거리 < (k,)
                eph_l_prev = anchor_eph_l[inds[j]] #j번째와 가까운 k개의 뭐야 eph
                update_rate = np.minimum(self.alpha * (1 - np.exp(-1 * dist**2 / self.std_dev_o)) + self.beta, self.alpha) # Eq. 5 
                eph_l_new = eph_l_prev * update_rate / (
                    eph_l_prev * update_rate + (1 - eph_l_prev) * (1 - update_rate)
                )
                #occupied update rule
                anchor_eph_l[inds[j]] = eph_l_new

            # free space update
            shifted_scan = scan - pose # local coordinates, 원점으로 옮긴거임
            sample_ratios = np.linspace(p_dor["min_ratio"], p_dor["max_ratio"], p_dor["num_samples"])
            #min_ratio부터 max_ratio까지 균등간격으로 num_samples개의 값 생성
            free_space_samples = pose + shifted_scan[:, np.newaxis, :] * sample_ratios.T[np.newaxis, :, np.newaxis]
            #shifted_scan:(N_i, 1, 3) 과 sample_ratios (1, num_sample ,1)로 바꿔서 브로드캐스팅함. (N_i, num_sample, 3)의 행렬 생성
            #+pose로 월드좌표계 복원
            free_space_samples = free_space_samples.reshape(-1, 3) # (N_i*num_sample, 3): 샘플링한 더미 free space를 1D로 flatten
            free_space_samples_o3d = o3d.geometry.PointCloud()
            free_space_samples_o3d.points = o3d.utility.Vector3dVector(free_space_samples)#np.array->o3d
            free_space_samples_o3d = free_space_samples_o3d.voxel_down_sample(voxel_size=0.1)#N_i*num_sample을 M'개로 다운샘플링
            free_space_samples = np.asarray(free_space_samples_o3d.points)#o3d->np array(M', 3)
            dists, inds = anchor_kdtree.query(free_space_samples, k=p_dor["num_k"])
            #dists:(M', K) = inds 차원 같음
            for j in range(len(dists)):
                dist = dists[j] #M'중 j번째 포인트의 dist 갖고옴
                eph_l_prev = anchor_eph_l[inds[j]] #j번째와 가까운 K개의 eph
                update_rate = np.maximum(self.alpha * (1 + np.exp(-1 * dist**2 / self.std_dev_f)) - self.beta, self.alpha) # Eq. 5
                eph_l_new = eph_l_prev * update_rate / (
                    eph_l_prev * update_rate + (1 - eph_l_prev) * (1 - update_rate)
                )
                anchor_eph_l[inds[j]] = eph_l_new

        # 3) 앵커에서 근처점으로 eph 전달
        distances, indices = anchor_kdtree.query(np.asarray(session_map.points), k=p_dor["num_k"])
        #distances: (M, K), indices: (M, K), M: 세션 맵 전체를 합치고 voxel grid 1cm로 다운샘플링
        distances = np.maximum(distances, 1e-6) # (M,), 포인트별 앵커점과 제일 큰거리 뽑음
        weights = 1 / (distances**2) #1/거리^2를 가중치로둠. 멀리잇을수록 덜퍼짐. (M,)
        weights /= np.sum(weights, axis=1, keepdims=True) #axis = 1로 정규화 (M,)
        eph_l = np.sum(weights * anchor_eph_l[indices], axis=1) #(M, K) * weights로 가중합: (M,)
        eph_l = np.clip(eph_l, 0, 1) # redundant, but for safety

        # 4) Remove dynamic objects to create cleaned session map
        static_points = session_map.select_by_index(np.where(eph_l <= p_dor["dynamic_threshold"])[0])
        static_eph_l = eph_l[eph_l <= p_dor["dynamic_threshold"]]
        static_points.paint_uniform_color([0.5, 0.5, 0.5])
        dynamic_points = session_map.select_by_index(np.where(eph_l > p_dor["dynamic_threshold"])[0])
        dynamic_points.paint_uniform_color([1, 0, 0])
                  
        if p_dor["save_static_dynamic_map"]:
            o3d.io.write_point_cloud(os.path.join(p_settings["output_dir"], "static_points.pcd"), static_points)  
            o3d.io.write_point_cloud(os.path.join(p_settings["output_dir"], "dynamic_points.pcd"), dynamic_points)
        if p_dor["viz_static_dynamic_map"]:
            total_points = static_points + dynamic_points
            o3d.visualization.draw_geometries([total_points])

        cleaned_session_map = SessionMap(
            np.asarray(static_points.points), static_eph_l
        )
        self.session_map = cleaned_session_map

        if p_dor["save_cleaned_session_map"]:
            cleaned_session_map.save(p_settings["output_dir"], is_global=False) 
        if p_dor["viz_cleaned_session_map"]:
            cleaned_session_map.visualize()

        return cleaned_session_map


    def get(self):
        return self.session_map
        

# Example usage
if __name__ == "__main__":
    config = "../config/sample.yaml"
    remover = MapRemover(config)
    # Load session using the config file or from an alingment module
    remover.load()
    # Run the dynamic object removal
    remover.run()
    # Get the cleaned session map
    cleaned_session_map = remover.get()