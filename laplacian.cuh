#pragma once

__device__
float uniform(int n, int seed, int t)
{
    curandStatePhilox4_32_10_t s;

    // seed a random number generator
    curand_init(seed,n, t, &s);

    float x = curand_uniform(&s);

    return x;
}



// (sheared) periodic boundary conditions
__device__ inline int idx(int i, int j, int N) {

    int shift=0;
    if(j==N)
      return ((i+N-shift) % N) + ((j+N) % N) * N;
    else if(j==-1)
      return ((i+N+shift) % N) + ((j+N) % N) * N;
    else
      return ((i+N) % N) + ((j+N) % N) * N;
}

__device__ inline float wrap_2pi(float theta) {
    theta = fmodf(theta, 2.0f * M_PI);
    return (theta < 0.0f) ? theta + 2.0f * M_PI : theta;
}

__global__ void apply_interactions(float* __restrict__ theta, float* __restrict__ lap, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= N || j >= N) return;

    float theta_c = theta[idx(i,j,N)];

    // antiperiodic bc in x, periodic shifted in y
    if(i==N-1)
    lap[i + j * N] =
      -sin(theta[idx(i+1,j,N)]-theta_c)+
      sin(theta[idx(i-1,j,N)]-theta_c)+
      sin(theta[idx(i,j+1,N)]-theta_c)+
      sin(theta[idx(i,j-1,N)]-theta_c);
    else if(i==0)
    lap[i + j * N] =
      sin(theta[idx(i+1,j,N)]-theta_c)+
      -sin(theta[idx(i-1,j,N)]-theta_c)+
      sin(theta[idx(i,j+1,N)]-theta_c)+
      sin(theta[idx(i,j-1,N)]-theta_c);
    else
    lap[i + j * N] =
      sin(theta[idx(i+1,j,N)]-theta_c)+
      sin(theta[idx(i-1,j,N)]-theta_c)+
      sin(theta[idx(i,j+1,N)]-theta_c)+
      sin(theta[idx(i,j-1,N)]-theta_c);
}

__global__ void apply_laplacian(float* __restrict__ theta, float* __restrict__ lap, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= N || j >= N) return;

    lap[i + j * N] =
        theta[idx(i+1,j,N)] + theta[idx(i-1,j,N)] +
        theta[idx(i,j+1,N)] + theta[idx(i,j-1,N)] -
        4.0f * theta[idx(i,j,N)];

    double shift = 0;

    // antiperiodic boundary conditions in x (i), periodic in y (j)
    if(i==N-1)
        lap[i + j * N] =
        wrap_2pi(theta[idx(i+1,j,N)]+1.0*M_PI) + theta[idx(i-1,j,N)] +
        theta[idx(i,j+1,N)] + theta[idx(i,j-1,N)] -
        4.0f * theta[idx(i,j,N)] + shift;

    if(i==0)
        lap[i + j * N] =
        theta[idx(i+1,j,N)] + wrap_2pi(theta[idx(i-1,j,N)]+1.0*M_PI) +
        theta[idx(i,j+1,N)] + theta[idx(i,j-1,N)] -
        4.0f * theta[idx(i,j,N)] - shift;

    float wrapped = wrap_2pi(theta[idx(i,j,N)]);
    theta[i + j * N] = wrapped;
}

