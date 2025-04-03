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

    __device__ unsigned char diff(unsigned char a, unsigned char b)
    {
        if (a > b)
        {
            return a - b;
        }
        return b-a;
    }

    __global__ void compute_sharpness(unsigned char* y_plane, unsigned char* s_plane, int width, int height, int pitch)
    {
        int x = blockIdx.x * blockDim.x + threadIdx.x;
        int y = blockIdx.y * blockDim.y + threadIdx.y;

        if ((x < width) && (y < height)) 
        {
            uint32_t ix = x + y*pitch;
            if( x > 0)
            {
                s_plane[ix] = diff(y_plane[ix - 1],  y_plane[ix]);
            }
            else
            {
                s_plane[ix] = diff(y_plane[ix + 1],  y_plane[ix]);
            }
        }
    }
  
  int cuda_compute_sharpness(void* y_ptr, void* s_ptr, uint16_t width, uint16_t height, uint16_t pitch)
  {
      cudaError_t err; 
      if (y_ptr == NULL) {
          fprintf(stderr, "Error: Null CUDA pointer A\n");
          return 0;
      }
      if (s_ptr == NULL) {
          fprintf(stderr, "Error: Null CUDA pointer B\n");
          return 0;
      }
      
      dim3 block(16,16);
      const dim3 grid(
          (width + block.x - 1) / block.x,
          (height + block.y - 1) / block.y
      );
      compute_sharpness<<<grid, block>>>((unsigned char* )y_ptr, (unsigned char*)s_ptr, width, height, pitch);
      err = cudaGetLastError();
      if (err != cudaSuccess) {
          fprintf(stderr, "CUDA kernel launch error: %s\n", cudaGetErrorString(err));
          return 0;
      }
      
      CUDA_CHECK(cudaDeviceSynchronize());
      //printf("Kernel completed successfully\n");
      return 1;
  }



__global__ void draw_kernel(unsigned char* y_plane, int width, int height, int pitch)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if ((x < width) && (y < height)) 
    {
        int m_x = 1920;
        int m_y = 1080;
        int dx = m_x - x;
        int dy = m_y - y;
        int r = dx*dx + dy*dy;
        int inner = 300;
        int outer = 310;
        if( r > inner*inner and r < outer*outer)
        {
            y_plane[ x + y*pitch] = 255;
        }
    }
}

int cuda_process_frame(void* y_ptr, uint16_t width, uint16_t height, uint16_t pitch)
{
    cudaError_t err; 
    if (y_ptr == NULL) {
        fprintf(stderr, "Error: Null CUDA pointer\n");
        return 0;
    }
    
    dim3 block(16,16);
    const dim3 grid(
        (width + block.x - 1) / block.x,
        (height + block.y - 1) / block.y
    );
    draw_kernel<<<grid, block>>>((unsigned char* )y_ptr, width, height, pitch);
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA kernel launch error: %s\n", cudaGetErrorString(err));
        return 0;
    }
    
    CUDA_CHECK(cudaDeviceSynchronize());
    //printf("Kernel completed successfully\n");
    return 1;
}


