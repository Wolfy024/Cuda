#include <cuda_runtime.h>
#include <iostream>

#define N 4
#define K 4
#define M 4
#define TILE 2

__global__ void VecMult(float* A, float* B, float* C)
{
    int by = blockIdx.y;
    int bx = blockIdx.x;

    int ty = threadIdx.y;
    int tx = threadIdx.x;

    int i = blockDim.y * by + ty; //blockIndex X
    int j = blockDim.x * bx + tx; //blockIndex Y

    __shared__ float sh_A[TILE][TILE];
    __shared__ float sh_B[TILE][TILE];

    float value = 0;
    for(int phase = 0; phase < N/TILE; phase++){
        sh_A[ty][tx] = A[i * N + phase * TILE +  tx];
        sh_B[ty][tx] = B[(phase * TILE + ty) * N + j];
        __syncthreads();
        for(int k = 0; k < TILE; k++) value += sh_A[ty][k]  * sh_B[k][tx];
        __syncthreads();
        C[i * N + j] = value;

    }
}

int main(){
        float h_A[N][K]= {
        {1, 2, 3, 4},
        {4, 5, 6, 7},
        {7, 8, 9, 10},
        {1, 1, 1, 1}
    };
        float h_B[K][M] = {
        {1, 2, 3, 4},
        {1, 1, 1, 1},
        {2, 2, 2, 2},
        {3, 3, 3, 3}
    };
        float h_C[N][M] = {0};

    float * d_A;
    float * d_B;
    float * d_C;

    cudaMalloc(&d_A, N * K * sizeof(float));
    cudaMalloc(&d_B, K * M * sizeof(float));
    cudaMalloc(&d_C, N * M * sizeof(float));

    cudaMemcpy(d_A, h_A, N * K * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, K * M * sizeof(float), cudaMemcpyHostToDevice);

    dim3 ThreadsPerBlock(TILE , TILE);
    dim3 blocks(
        (M + TILE - 1) / TILE,
        (N + TILE - 1) / TILE
    );

    VecMult<<<blocks, ThreadsPerBlock>>>(d_A, d_B, d_C);
    cudaDeviceSynchronize();

    cudaMemcpy(h_C, d_C, N * M * sizeof(float), cudaMemcpyDeviceToHost);

    std::cout << "Results: \n";
    for(int i = 0; i < N; i++){
        for(int j = 0; j < M; j++){
            std::cout << h_C[i][j] << " ";
        }
        std::cout << "\n";
    }
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    return 0;
}



//  threadIdx.x = {0, 1}
//  threadIdx.y = {0, 1}
//  blockIdx.x = {0, 1}
//  blockIdx.y = {0, 1}
// Tiles = {0, 1}


// Col traversal B[(t * TILE + threadIdx.y) * N + col]; -> HOW
// Row traversal A[row * N + t * TILE + threadIdx.x]; -> HOW

// N = 8, M = 8, K = 8
// GridSize(2, 2)
// Blocks(4, 4)

// sh_A[ty][tx] = A[i * N + phase * TILE + tx]; -> Understood.
// i * N = row
// phase * TILE = Tile
// tx = particular element [0, 1]


// sh_B[ty][tx] = B[(phase * TILE + ty) * N + j]; -> 



// i * N = row (1 * 3) = 3
// j = col 
// final_indexing i * N + j (3 + 1) = 4