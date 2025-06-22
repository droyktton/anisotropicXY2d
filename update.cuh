#pragma once

__global__ void update_theta(const float* __restrict__ theta,
                             float* __restrict__ theta_new,
                             const float* __restrict__ lap,
                             const float* __restrict__ Hms,
                             float J, float K, float dt, int N, float H, float Hang, int nt) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= N || j >= N) return;

    int id = i + j * N;
    float th = theta[id];

    float rn = uniform(id, 1234, nt);
    rn = (rn - 0.5f)*sqrt(0.1)/sqrtf(dt);

    theta_new[id] = th + dt * (J * lap[id] + K * sinf(2 * th + M_PI) + H*sinf(Hang-th) + 0.0*Hms[id] + rn);
}
