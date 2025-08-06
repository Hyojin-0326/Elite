#include <cuda_runtime.h>

// --- Occupied Batch Kernel ---
__global__ void occ_batch_kernel(const float* scans, const float* poses,
                                 const float* anchors,
                                 int S, int N, int M, int K,
                                 float alpha, float beta, float sigma_o,
                                 float* logits, float* counts) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = S * N;
    if (idx >= total) return;

    int i = idx / N;  // scan index
    int j = idx % N;  // point index

    // 포인터 연산: current point
    const float* pt = scans + (i * N + j) * 3;

    // TODO: 여기에 KNN → top-K indices(ids[K]) + distances(dists[K]) 계산 로직 삽입
    // 예시 변수:
    // int ids[16];
    // float dists[16];

    // 업데이트
    for (int t = 0; t < K; ++t) {
        float d = dists[t];
        float rate = fminf(alpha * (1.0f - expf(-d*d/(sigma_o*sigma_o))) + beta, alpha);
        float lg   = logf(rate / (1.0f - rate + 1e-9f));
        int   aidx = ids[t];
        atomicAdd(&logits[aidx], lg);
        atomicAdd(&counts[aidx], 1.0f);
    }
}

// --- Free Batch Kernel ---
__global__ void free_batch_kernel(const float* scans, const float* poses,
                                  const float* anchors,
                                  int S, int N, int M, int K,
                                  float alpha, float beta, float sigma_f,
                                  float* logits, float* counts) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = S * N * K;  // sample 갯수(포인트마다 K개)
    if (idx >= total) return;

    int i = idx / (N * K);       // scan index
    int rem = idx % (N * K);
    int j = rem / K;             // point index
    int s = rem % K;             // sample index

    // sample point 계산
    const float* scan_pt = scans + (i * N + j) * 3;
    const float* pose   = poses + i * 3;
    float sample_pt[3];
    float ratio = /* sample ratio array[s] */;  // 미리 상수 또는 GPU메모리에 올려둬야 함
    for (int c = 0; c < 3; ++c)
        sample_pt[c] = pose[c] + (scan_pt[c] - pose[c]) * ratio;

    // TODO: KNN → ids[K], dists[K]

    for (int t = 0; t < K; ++t) {
        float d = dists[t];
        float rate = fmaxf(alpha * (1.0f + expf(-d*d/(sigma_f*sigma_f))) - beta, alpha);
        float lg   = logf(rate / (1.0f - rate + 1e-9f));
        int   aidx = ids[t];
        atomicAdd(&logits[aidx], lg);
        atomicAdd(&counts[aidx], 1.0f);
    }
}

// --- 런치 헬퍼 ---
void launch_occ_batch(const float* scans, const float* poses,
                      const float* anchors,
                      int S, int N, int M, int K,
                      float alpha, float beta, float sigma_o,
                      float* logits, float* counts) {
    int total = S * N;
    int threads = 256;
    int blocks  = (total + threads - 1) / threads;
    occ_batch_kernel<<<blocks, threads>>>(
        scans, poses, anchors, S, N, M, K,
        alpha, beta, sigma_o,
        logits, counts
    );
}

void launch_free_batch(const float* scans, const float* poses,
                       const float* anchors,
                       int S, int N, int M, int K,
                       float alpha, float beta, float sigma_f,
                       float* logits, float* counts) {
    int total = S * N * K;
    int threads = 256;
    int blocks  = (total + threads - 1) / threads;
    free_batch_kernel<<<blocks, threads>>>(
        scans, poses, anchors, S, N, M, K,
        alpha, beta, sigma_f,
        logits, counts
    );
}
