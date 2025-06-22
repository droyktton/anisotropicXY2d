#include <thrust/device_vector.h>
#include <thrust/transform.h>
#include <cufft.h>
#include <curand_kernel.h>
#include <cmath>
#include <iostream>
#include <fstream>
#include <iomanip>  // for std::setprecision

/* counter-based random numbers in curand API */
#include <curand_kernel.h>


#include "laplacian.cuh"
#include "update.cuh"
#include "magnetostatics.cuh"

constexpr int N = 256;
constexpr int STEPS = 10000;
constexpr float J = 1.0f;
constexpr float K = 0.5f;
constexpr float dt = 0.01f;
constexpr float H = 1.0f;
constexpr float Hang = 0.0f;


__global__ void init_domain_wall(float* theta) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= N || j >= N) return;

    int idx = i + j * N;
    float x = i - N / 2;
    theta[idx] = 2.f*atan(exp(x / 10.0f))*1.0;  // domain wall along x
}

void save_theta_csv(const std::vector<float>& theta, int N, const std::string& filename) {
    std::ofstream out(filename);
    for (int j = 0; j < N; ++j) {
        for (int i = 0; i < N; ++i) {
            out << theta[i + j * N];
            if (i < N - 1) out << " ";
        }
        out << "\n";
    }
}

int main() {
    thrust::device_vector<float> theta(N * N);
    thrust::device_vector<float> theta_new(N * N);
    thrust::device_vector<float> lap(N * N);
    thrust::device_vector<float> Hms(N * N, 0.0f);

    dim3 threads(16, 16);
    dim3 blocks((N + 15)/16, (N + 15)/16);
    init_domain_wall<<<blocks, threads>>>(thrust::raw_pointer_cast(theta.data()));
    cudaDeviceSynchronize();

    // GPU → CPU
    std::vector<float> theta_h(N * N);

    for (int step = 0; step < STEPS; ++step) {

        if(step%100==0) {
          std::cout << "\r" << std::setprecision(5) << step*100./STEPS << "%" << std::flush;
          thrust::copy(theta.begin(), theta.end(), theta_h.begin());
          // Save to file
          std::string filename = "theta_" + std::to_string(step) + ".csv";
          save_theta_csv(theta_h, N, filename);
        }

        //apply_laplacian<<<blocks, threads>>>(thrust::raw_pointer_cast(theta.data()), thrust::raw_pointer_cast(lap.data()), N);
        apply_interactions<<<blocks, threads>>>(thrust::raw_pointer_cast(theta.data()), thrust::raw_pointer_cast(lap.data()), N);

        /*cudaDeviceSynchronize();
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess)
            std::cerr << "CUDA error: " << cudaGetErrorString(err) << std::endl;
        */


        //compute_magnetostatics(thrust::raw_pointer_cast(theta.data()), thrust::raw_pointer_cast(Hms.data()), N);

        /*cudaDeviceSynchronize();
        err = cudaGetLastError();
        if (err != cudaSuccess)
            std::cerr << "CUDA error: " << cudaGetErrorString(err) << std::endl;
        */


        update_theta<<<blocks, threads>>>(thrust::raw_pointer_cast(theta.data()),
                                          thrust::raw_pointer_cast(theta_new.data()),
                                          thrust::raw_pointer_cast(lap.data()),
                                          thrust::raw_pointer_cast(Hms.data()),
                                          J, K, dt, N, H, Hang, step);
        /*cudaDeviceSynchronize();
        err = cudaGetLastError();
        if (err != cudaSuccess)
            std::cerr << "CUDA error: " << cudaGetErrorString(err) << std::endl;
        */

        theta.swap(theta_new);
    }




    std::cout << "\nSimulation completed.\n";
    return 0;
}
