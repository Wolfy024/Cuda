#include <cuda_runtime.h>
#include <iostream>

#define row_A 4
#define col_A_row_B 3
#define col_B 5

__global__ void VecMult(float*A, float*B, float*C, int ro_A, int co_A_ro_B, int co_B)
    {
        int row = blockIdx.y * blockDim.y + threadIdx.y;
        int col = blockIdx.x * blockDim.x + threadIdx.x;
        if (row < row_A && col < col_B)
        {
            float sum = 0.0f;
            for(int i = 0; i < co_A_ro_B; i++) sum += A[row * co_A_ro_B + i] * B[i * co_B + col];
            C[row * co_B + col] = sum;
        }
    }

int main(){
        float h_A[row_A][col_A_row_B]= {
        {1, 2, 3},
        {4, 5, 6},
        {7, 8, 9},
        {1, 1, 1}
    };
        float h_B[col_A_row_B][col_B] = {
        {1, 2, 3, 4, 5},
        {1, 1, 1, 1, 1},
        {2, 2, 2, 2, 2}
    };
        float h_C[row_A][col_B] = {0};

    float * d_A;
    float * d_B;
    float * d_C;

    cudaMalloc(&d_A, row_A * col_A_row_B * sizeof(float));
    cudaMalloc(&d_B, col_A_row_B * col_B * sizeof(float));
    cudaMalloc(&d_C, row_A * col_B * sizeof(float));

    cudaMemcpy(d_A, h_A, row_A * col_A_row_B * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, col_A_row_B * col_B * sizeof(float), cudaMemcpyHostToDevice);

    dim3 ThreadsPerBlock(16, 16);
    dim3 blocks(
        (col_B + ThreadsPerBlock.x - 1)/ThreadsPerBlock.x,
        (row_A + ThreadsPerBlock.y - 1)/ThreadsPerBlock.y
    );
    VecMult<<<blocks, ThreadsPerBlock>>>(d_A, d_B, d_C, row_A, col_A_row_B, col_B);
    cudaDeviceSynchronize();

    cudaMemcpy(h_C, d_C, row_A * col_B * sizeof(float), cudaMemcpyDeviceToHost);

    std::cout << "Results: \n";
    for(int i = 0; i < row_A; i++){
        for(int j = 0; j < col_B; j++){
            std::cout << h_C[i][j] << " ";
        }
        std::cout << "\n";
    }
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    return 0;
}