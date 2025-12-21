#include <cuda_runtime.h>
#include <iostream>
#include <random>

#define M 2048
#define K 2048
#define N 2048
#define TILES 16

__global__ void matMul(float *A, float *B, float *C)
{
    __shared__ float As[TILES][TILES];
    __shared__ float Bs[TILES * 2][TILES + 1];

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int row  = blockIdx.y * blockDim.y + ty;
    int col0 = blockIdx.x * blockDim.x * 2 + tx;
    int col1 = col0 + TILES;

    float acc0 = 0.0f;
    float acc1 = 0.0f;

    int phases = (K + TILES - 1) / TILES;
    for (int phase = 0; phase < phases; phase++)
    {
        int col_A = phase * TILES + tx;
        As[ty][tx] = (row < M && col_A < K)
                         ? A[row * K + col_A]
                         : 0.0f;

        int row_B = phase * TILES + ty;
        Bs[tx][ty] = (row_B < K && col0 < N)
                         ? B[row_B * N + col0]
                         : 0.0f;

        Bs[tx + TILES][ty] = (row_B < K && col1 < N)
                                 ? B[row_B * N + col1]
                                 : 0.0f;

        __syncthreads();

        for (int i = 0; i < TILES; i++)
        {
            float a = As[ty][i];
            acc0 += a * Bs[i][tx];
            acc1 += a * Bs[i][tx + TILES];
        }

        __syncthreads();
    }

    if (row < M)
    {
        if (col0 < N)
            C[row * N + col0] = acc0;
        if (col1 < N)
            C[row * N + col1] = acc1;
    }
}

int main()
{
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<int> dist(1, 9);

    float *h_A = new float[M * K];
    float *h_B = new float[K * N];
    float *h_C = new float[M * N];

    for (int i = 0; i < M * K; i++)
        h_A[i] = dist(gen);

    for (int i = 0; i < K * N; i++)
        h_B[i] = dist(gen);

    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, M * K * sizeof(float));
    cudaMalloc(&d_B, K * N * sizeof(float));
    cudaMalloc(&d_C, M * N * sizeof(float));

    cudaMemcpy(d_A, h_A, M * K * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, K * N * sizeof(float), cudaMemcpyHostToDevice);

    dim3 threads(TILES, TILES);
    dim3 blocks(
        (N + TILES*2 - 1) / (TILES*2),
        (M + TILES - 1) / TILES
    );

    matMul<<<blocks, threads>>>(d_A, d_B, d_C);
    cudaDeviceSynchronize();

    cudaMemcpy(h_C, d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost);

    std::cout << "Results:";
    for(int i = 0; i < M * N; i++)
    {
        if (i % N == 0) std::cout << "\n";
        std::cout << h_C[i] << " ";
    }
    std::cout << '\n';
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}
