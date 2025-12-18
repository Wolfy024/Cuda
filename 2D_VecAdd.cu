#include <cuda_runtime.h>
#include <iostream>
#define N 10

__global__ void vecAdd(float*A, float*B, float*C, int n){
    int col = blockDim.x * blockIdx.x + threadIdx.x;
    int row = blockDim.y * blockIdx.y + threadIdx.y;
    if (row < 10 && col < 10){
        C[row * n + col] = A[row * n + col] + B[row * n + col];
    }
}

int main(){
    float h_A[N][N];
    float h_B[N][N];
    float h_C[N * N];
    for(int row = 0; row < 10; row++){
        for(int col = 0; col < 10; col++){
            h_A[row][col] = row + col;
            h_B[row][col] = (row + col) * 10;
        }
    }

    float * d_A;
    float * d_B;
    float * d_C;
    cudaMalloc(&d_A, N*N*sizeof(float));
    cudaMalloc(&d_B, N*N*sizeof(float));
    cudaMalloc(&d_C, N*N*sizeof(float));

    cudaMemcpy(d_A, h_A, N*N*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, N*N*sizeof(float), cudaMemcpyHostToDevice);

    dim3 threadsPerBlock(16, 16);
    dim3 blocks((N + 15) / 16, (N + 15) / 16);
    vecAdd<<<blocks, threadsPerBlock>>>(d_A, d_B, d_C, N);
    cudaDeviceSynchronize();
    cudaMemcpy(h_C, d_C, N*N*sizeof(float), cudaMemcpyDeviceToHost);

    std::cout << "Results:\n";
    for(int i = 0; i <  10; i++){
        for(int j = 0; j < 10; j++){
            std::cout<< h_C[i*N + j];
            std::cout<< ' ';
        }
        std::cout << '\n';
    }
    return 0;
}