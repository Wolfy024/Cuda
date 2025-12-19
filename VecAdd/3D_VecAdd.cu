#include <cuda_runtime.h>
#include <iostream>
#define rowNum 10
#define colNum 10
#define zNum 10
__global__ void vecAdd(float* A, float*B, float*C, int X, int Y, int Z){
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;

    if (x < X && y < Y && z < Z){
        int idx = x * (Y * Z) + y * Z + z;
        C[idx] = A[idx] + B[idx];
    }
}

int main(){
    float h_A[rowNum][colNum][zNum];
    float h_B[rowNum][colNum][zNum];
    float h_C[rowNum][colNum][zNum];

    for (int i = 0; i < 10; i++){
        for (int j = 0; j < 10; j++){
            for (int k = 0; k<10; k++){
                h_A[i][j][k] = i + j + k;
                h_B[i][j][k] = (i + j + k) * 10;
            }
        }
    }
    float * d_A;
    float * d_B;
    float * d_C;
    cudaMalloc(&d_A, rowNum * colNum * zNum * sizeof(float));
    cudaMalloc(&d_B, rowNum * colNum * zNum * sizeof(float));
    cudaMalloc(&d_C, rowNum * colNum * zNum * sizeof(float));
    
    cudaMemcpy(d_A, h_A, rowNum * colNum * zNum * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, rowNum * colNum * zNum * sizeof(float), cudaMemcpyHostToDevice);

    dim3 threadsPerBlock(8, 8, 8);
    dim3 block(
    (rowNum + threadsPerBlock.x - 1) / threadsPerBlock.x,
    (colNum + threadsPerBlock.y - 1) / threadsPerBlock.y,
    (zNum   + threadsPerBlock.z - 1) / threadsPerBlock.z
);
    vecAdd<<<block, threadsPerBlock>>>(d_A, d_B, d_C, rowNum, colNum, zNum);
    cudaDeviceSynchronize();
    cudaMemcpy(h_C, d_C, rowNum * colNum * zNum * sizeof(float), cudaMemcpyDeviceToHost);

    std::cout << "Results: \n";
    for(int i = 0; i < 10; i++){
        for (int j = 0; j < 10; j++){
            for (int k = 0; k<10; k++){
                std::cout << h_C[i][j][k] << ' ';
            }
            std::cout << '\n';
        }
        std::cout << '\n';
    }
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    return 0;

}