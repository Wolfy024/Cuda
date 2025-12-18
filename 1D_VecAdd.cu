#include <cuda_runtime.h>
#include <iostream>
#define N 10


// grid -> block -> thread

__global__ void VecAdd(float* A, float* B, float* C){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N){
        C[i] = A[i] + B[i];
    }
}

int main(){
    float h_A[N], h_B[N], h_C[N];
    
    for (int i = 0; i < N; i++){
        h_A[i] = i;
        h_B[i] = i * 10;
    }
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, N * sizeof(float));
    cudaMalloc(&d_B, N * sizeof(float));
    cudaMalloc(&d_C, N * sizeof(float));

    cudaMemcpy(d_A, h_A, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, N * sizeof(float), cudaMemcpyHostToDevice);

    int threadPerBlock = 256;
    int blocks = (N + threadPerBlock - 1) / threadPerBlock;
    VecAdd<<<blocks, threadPerBlock>>>(d_A, d_B, d_C);

    cudaDeviceSynchronize();

    cudaMemcpy(h_C, d_C, N * sizeof(float), cudaMemcpyDeviceToHost);
    std::cout << "Results: \n";
    for (int i = 0; i < N; i++) {
        std::cout << h_C[i] << " ";
    }
    std::cout << "\n";
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    return 0;
}