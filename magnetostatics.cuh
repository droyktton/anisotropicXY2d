
#pragma once
#include <cufft.h>
#include <thrust/complex.h>
#include <thrust/device_vector.h>

__global__ void compute_divergence(const float* theta, float* rho, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= N || j >= N) return;

    int id = i + j * N;

    float dmx = cosf(theta[idx((i+1)%N,j,N)]) - cosf(theta[idx((i-1+N)%N,j,N)]);
    float dmy = sinf(theta[idx(i,(j+1)%N,N)]) - sinf(theta[idx(i,(j-1+N)%N,N)]);
    rho[id] = 0.5f * (dmx + dmy);
}

void compute_magnetostatics(const float* theta, float* Hms, int N) {
    static thrust::device_vector<float> rho(N * N);
    static thrust::device_vector<thrust::complex<float>> rho_k(N * (N/2 + 1));
    static thrust::device_vector<thrust::complex<float>> phi_k(N * (N/2 + 1));
    static cufftHandle planR2C, planC2R;
    static bool initialized = false;

    if (!initialized) {
        cufftPlan2d(&planR2C, N, N, CUFFT_R2C);
        cufftPlan2d(&planC2R, N, N, CUFFT_C2R);
        initialized = true;
    }

    dim3 threads(16, 16);
    dim3 blocks((N + 15)/16, (N + 15)/16);
    compute_divergence<<<blocks, threads>>>(theta, thrust::raw_pointer_cast(rho.data()), N);
    cudaDeviceSynchronize();

    std::cout << "divergence OK" << std::endl;

    cufftExecR2C(planR2C, thrust::raw_pointer_cast(rho.data()),
                 (cufftComplex*)thrust::raw_pointer_cast(rho_k.data()));

    std::cout << "R2C OK" << std::endl;

    auto rk = thrust::raw_pointer_cast(rho_k.data());
    auto pk = thrust::raw_pointer_cast(phi_k.data());
    int halfN = N / 2 + 1;

    for (int ky = 0; ky < N; ++ky) {
        for (int kx = 0; kx < halfN; ++kx) {
            int idx = kx + ky * halfN;
            float kx_ = sinf(M_PI * kx / N);
            float ky_ = sinf(M_PI * ky / N);
            float k2 = 4.0f * (kx_ * kx_ + ky_ * ky_);
            pk[idx] = (k2 > 0.0f) ? rk[idx] / k2 : thrust::complex<float>(0, 0);
        }
    }
    std::cout << "wavevectors OK" << std::endl;

    cufftExecC2R(planC2R, (cufftComplex*)thrust::raw_pointer_cast(phi_k.data()), Hms);
    std::cout << "C2R OK" << std::endl;


    thrust::transform(
      Hms, Hms + N * N,
      Hms,
      [=] __device__ (float x) { return x / (N * N); }
    );

}
