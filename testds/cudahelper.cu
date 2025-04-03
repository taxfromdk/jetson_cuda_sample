#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cuda_runtime_api.h>
#include "cudahelper.h"

// Macro for CUDA error checking
#define CUDA_CHECK(call) \
  do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
      fprintf(stderr, "CUDA error in %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
      return 0; \
    } \
  } while(0)


#define BLOCK_LINEAR_WIDTH 64
#define BLOCK_LINEAR_HEIGHT 16

__device__ size_t block_linear_offset(size_t x, size_t y, size_t pitch)
{
    size_t block_x = x / BLOCK_LINEAR_WIDTH;
    size_t block_y = y / BLOCK_LINEAR_HEIGHT;

    size_t inner_x = x % BLOCK_LINEAR_WIDTH;
    size_t inner_y = y % BLOCK_LINEAR_HEIGHT;

    size_t block_offset = (block_y * (pitch / BLOCK_LINEAR_WIDTH) + block_x) * (BLOCK_LINEAR_WIDTH * BLOCK_LINEAR_HEIGHT);
    size_t pixel_offset = inner_y * BLOCK_LINEAR_WIDTH + inner_x;

    return block_offset + pixel_offset;
}

__global__ void draw_square_kernel(unsigned char* y_plane, int width, int height, int pitch)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if ((x < width) && (y < height)) 
    {
        if(y > 800)
        {
            unsigned char* pixel = y_plane + block_linear_offset(x,y,pitch);
            *pixel = 255;
        }
    }
}

int cuda_process_frame(int gpuId, void* y_ptr, uint16_t width, uint16_t height, uint16_t pitch)
{
    cudaError_t err; 
    printf("cuda_process_frame started\n");
    
    if (y_ptr == NULL) {
        fprintf(stderr, "Error: Null CUDA pointer\n");
        return 0;
    }
    
    // Set CUDA device
    //CUDA_CHECK(cudaSetDevice(gpuId));
    
    dim3 block(8,8);
    dim3 grid(480, 270);
    draw_square_kernel<<<grid, block>>>((unsigned char* )y_ptr, width, height, pitch);
    printf("What a day\n");
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA kernel launch error: %s\n", cudaGetErrorString(err));
        return 0;
    }
    
    CUDA_CHECK(cudaDeviceSynchronize());

    printf("Empty kernel completed successfully\n");
    return 1;
}