#include <cuda_runtime.h>
#include <iostream>
#include <random>

#define N 4
#define K 4
#define M 4

#define TILES 2


// A weird case. 
// Bank Conflicts. Transposed matrix mult introduced Bank conflicts which got fixed by padding.
// But due to additional operations in this version. It's actually slower than normal version by about 2x.
// Probably will get to know more, once we proceed.

__global__ void matMulC(float * A, float * B, float * C){

    __shared__ float As[TILES][TILES];
    __shared__ float Bs[TILES][TILES + 1];

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int dx = blockIdx.x;
    int dy = blockIdx.y;

    int row = blockDim.y * dy + ty;
    int col = blockDim.x * dx + tx;

    int phases = (N + TILES - 1) / TILES;
    float val = 0;
    for (int phase = 0; phase < phases; phase++){
        int col_A = phase * TILES + tx;
        if (col_A < K && row < M){
            As[ty][tx] = A[row * K + col_A];
        }
        else{
            As[ty][tx] = 0.0f;
        }
        int row_B = phase * TILES + ty;
        if (row_B < K && col < N){
            Bs[tx][ty] = B[row_B * N + col];
        }
        else{
            Bs[tx][ty] = 0.0f;
        }
        __syncthreads();
        for (int i = 0; i<TILES; i++){
            val += As[ty][i] * Bs[tx][i];
        }
    }
    if (row < M && col < N)
    {
        C[row * N + col] = val;
    }
}

int main(){
    // Random Number (Damn, what !?)
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution <int> dist(1, 9);

    float* h_A = new float[M * K];
    float* h_B = new float[K * N];
    float* h_C = new float[M * N];


    // Fill with random numbers
    for(int i = 0; i < M*K; i++){
        h_A[i] = dist(gen);
    }
    for(int i = 0; i < K*N; i++){
        h_B[i] = dist(gen);

    }

    float * d_A;
    float * d_B;
    float * d_C;

    cudaMalloc(&d_A, M * K * sizeof(float));
    cudaMalloc(&d_B, K * N * sizeof(float));
    cudaMalloc(&d_C, M * N * sizeof(float));

    cudaMemcpy(d_A, h_A, M * K * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, K * N  * sizeof(float), cudaMemcpyHostToDevice);
    dim3 ThreadsPerBlock(TILES, TILES);
    dim3 Blocks((N + TILES - 1)/ TILES, (M + TILES - 1)/ TILES);
    matMulC<<<Blocks, ThreadsPerBlock>>>(d_A, d_B, d_C);
    cudaDeviceSynchronize();
    cudaMemcpy(h_C, d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    
    std::cout << "Results:";
    for(int i = 0; i < M * N; i++)
    {
        if (i % N == 0) std::cout << "\n";
        std::cout << h_C[i] << " ";
    }
    std::cout << '\n';
}