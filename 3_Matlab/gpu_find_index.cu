#include "gpu_find_index.h"

__global__ void gpu_find_index_CUDA(bool* dst, double* origX, double* origY, double* thrX, double* thrY, int numRows_orig, int numRows_thr)
{
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if(row >= numRows_orig)
        return;
    for(int col = 0; col < numRows_thr; col++)
    {
        if (origX[row] == thrX[col])
        {
            if (origY[row] == thrY[col])
            {
                dst[row] = true;
            }
        }
    }
}
void gpu_find_index(bool* dst, double* origX, double* origY, double* thrX, double* thrY, int numRows_orig, int numRows_thr)
{
    double* device_origX, *device_origY, *device_thrX, *device_thrY;
    bool* device_dst;
    
    cudaMalloc(&device_origX, sizeof(double) * numRows_orig);
    cudaMalloc(&device_origY, sizeof(double) * numRows_orig);
    cudaMalloc(&device_thrX, sizeof(double) * numRows_thr);
    cudaMalloc(&device_thrY, sizeof(double) * numRows_thr);
    cudaMalloc(&device_dst, sizeof(bool) * numRows_orig);

    cudaMemcpy(device_origX, origX, sizeof(double) * numRows_orig, cudaMemcpyHostToDevice);
    cudaMemcpy(device_origY, origY, sizeof(double) * numRows_orig, cudaMemcpyHostToDevice);
    cudaMemcpy(device_thrX, thrX, sizeof(double) * numRows_thr, cudaMemcpyHostToDevice);
    cudaMemcpy(device_thrY, thrY, sizeof(double) * numRows_thr, cudaMemcpyHostToDevice);

    cudaMemset(device_dst, false, sizeof(bool) * numRows_orig);

    dim3 blockSize(16*16);     
    dim3 gridSize((numRows_orig+(16*16-1))/blockSize.x);
    
    gpu_find_index_CUDA<<<gridSize,blockSize>>>(device_dst, device_origX, device_origY, device_thrX, device_thrY, numRows_orig, numRows_thr);

    cudaMemcpy(dst, device_dst, sizeof(bool) * numRows_orig, cudaMemcpyDeviceToHost);

    cudaFree(device_origX);
    cudaFree(device_origY);
    cudaFree(device_thrX);
    cudaFree(device_thrY);
    cudaFree(device_dst);
}
