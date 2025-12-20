#include <cuda_runtime.h>
#include <iostream>
#include <random>

#define M 8 // Row of Matrix A
#define K 5 // Col of Matrix A, Row of Matrix B
#define N 8 // Col of Matrix B

#define TILE 2


__global__ void matMul(float * A, float * B, float * C)
{
    __shared__ float As[TILE][TILE];
    __shared__ float Bs[TILE][TILE];

    int by = blockIdx.y;
    int bx = blockIdx.x; 

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int row = blockDim.y * by + ty;
    int col = blockDim.x * bx + tx;

    float val = 0.0f;

    for(int phase = 0; phase < (K + TILE - 1)/TILE; phase++){
        
        //  As Fill
        int a_col = phase * TILE  + tx;
        if (row < M && a_col < K) 
        {
            As[ty][tx] = A[row * K + a_col];
        }
        else 
        {
            As[ty][tx] = 0.0f;
        }

        // Bs Fill
        int b_row = phase * TILE + ty;
        if (col < N && b_row < K)
        {
            Bs[ty][tx] = B[b_row * N + col];
        }
        else
        {
            Bs[ty][tx] = 0.0f;
        }
        __syncthreads();

        // Calculate
        for(int i = 0; i < TILE; i++){
            val += As[ty][i] * Bs[i][tx]; 
        }
        __syncthreads();
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

    float h_A[M][K];
    float h_B[K][N];
    float h_C[M][N];

    // Fill with random numbers
    for(int i = 0; i < M; i++){
        for (int j = 0; j < K; j++){
            h_A[i][j] = dist(gen);
        }
    }
    for(int i = 0; i < K; i++){
        for(int j = 0; j < N; j++){
            h_B[i][j] = dist(gen);
        }
    }

    float * d_A;
    float * d_B;
    float * d_C;

    cudaMalloc(&d_A, M * K * sizeof(float));
    cudaMalloc(&d_B, K * N * sizeof(float));
    cudaMalloc(&d_C, M * N * sizeof(float));

    cudaMemcpy(d_A, h_A, M * K * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, K * N  * sizeof(float), cudaMemcpyHostToDevice);

    dim3 ThreadsPerBlock(TILE, TILE);
    dim3 Blocks((N + TILE - 1)/ TILE, (M + TILE - 1)/ TILE);

    matMul<<<Blocks, ThreadsPerBlock>>>(d_A, d_B, d_C);
    cudaDeviceSynchronize();

    cudaMemcpy(h_C, d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    
    std::cout << "Results:\n";
    for(int i = 0; i < M; i++){
        for(int j = 0; j < N; j++){
            std::cout << h_C[i][j] << " ";;
        }
    std::cout << '\n';
    }
}