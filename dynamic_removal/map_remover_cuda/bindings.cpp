#include <torch/extension.h>
#include <vector>
#include <cuda_runtime.h>

// CUDA 커널 선언
void launch_occ_batch(const float* scans, const float* poses,
                      const float* anchors, int S, int N, int M, int K,
                      float alpha, float beta, float sigma_o,
                      float* logits, float* counts);
void launch_free_batch(const float* scans, const float* poses,
                       const float* anchors, int S, int N, int M, int K,
                       float alpha, float beta, float sigma_f,
                       float* logits, float* counts);

void batch_scan_simple(torch::Tensor scans,    // [S, N, 3]
                       torch::Tensor poses,    // [S,   3]
                       torch::Tensor anchors,  // [M,   3]
                       torch::Tensor logits,   // [M]
                       torch::Tensor counts,   // [M]
                       int K,
                       float alpha, float beta,
                       float sigma_o, float sigma_f) {
    // GPU 상 연속 메모리
    scans   = scans.contiguous();
    poses   = poses.contiguous();
    anchors = anchors.contiguous();
    logits  = logits.contiguous();
    counts  = counts.contiguous();

    int S = scans.size(0);
    int N = scans.size(1);
    int M = anchors.size(0);

    const float* scans_ptr  = scans.data_ptr<float>();
    const float* poses_ptr  = poses.data_ptr<float>();
    const float* anchor_ptr = anchors.data_ptr<float>();
    float*       logits_ptr = logits.data_ptr<float>();
    float*       counts_ptr = counts.data_ptr<float>();

    // 1) Occupied update 배치 처리
    launch_occ_batch(
        scans_ptr, poses_ptr, anchor_ptr,
        S, N, M, K, alpha, beta, sigma_o,
        logits_ptr, counts_ptr
    );

    // 2) Free update 배치 처리
    launch_free_batch(
        scans_ptr, poses_ptr, anchor_ptr,
        S, N, M, K, alpha, beta, sigma_f,
        logits_ptr, counts_ptr
    );
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("batch_scan_simple", &batch_scan_simple,
          "Simple batch scan update");
}
