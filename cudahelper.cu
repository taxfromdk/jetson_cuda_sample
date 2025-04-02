 #include <stdio.h>
 #include <stdlib.h>
 #include <string.h>
 #include <cuda_runtime.h>
 #include <cuda_runtime_api.h>
 #include "cudahelper.h"
 
 // Macro for CUDA error checking - useful for debugging
 #define CUDA_CHECK(call) \
   do { \
     cudaError_t err = call; \
     if (err != cudaSuccess) { \
       fprintf(stderr, "CUDA error in %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
       return 0; \
     } \
   } while(0)
 
 /* CUDA kernel for drawing text on NV12 frame */
 __global__ void drawKernel(
     unsigned char* d_frame,
     int pitch,
     int width,
     int height)
 {
     int x = blockIdx.x * blockDim.x + threadIdx.x;
     int y = blockIdx.y * blockDim.y + threadIdx.y;
     
     if (x >= width || y >= height) return;
     
     if (x < 50 && y < 50) {
         d_frame[y * pitch + x] = 255;
     }
 }
 
 int cuda_process_frame(int gpuId, void* y_plane, uint16_t width, uint16_t height, uint16_t pitch)
 {
    cudaError_t err; 
    printf("cuda_process_frame started\n");
     if (y_plane == NULL) {
         fprintf(stderr, "Error: Null CUDA pointer\n");
         return 0;
     }
     
     CUDA_CHECK(cudaSetDevice(gpuId));
     
     // Clear any previous errors
     err = cudaGetLastError();
     if (err != cudaSuccess) {
         printf("CUDA context error before processing: %s\n", cudaGetErrorString(err));
     }
     printf("Processing frame with CUDA: width=%d, height=%d, pitch=%d, ptr=%p\n", 
            width, height, pitch, y_plane);
     
    CUDA_CHECK(cudaDeviceSynchronize());
     

    dim3 threadsPerBlock(8, 8);
    dim3 numBlocks(
        (width + threadsPerBlock.x - 1) / threadsPerBlock.x,
        (height + threadsPerBlock.y - 1) / threadsPerBlock.y
    );
    
    printf("Launching kernel with grid: (%d,%d), block: (%d,%d)\n", 
            numBlocks.x, numBlocks.y, threadsPerBlock.x, threadsPerBlock.y);
    
    drawKernel<<<numBlocks, threadsPerBlock>>>(
        (unsigned char*)y_plane,
        pitch,
        width,
        height
    );
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA kernel launch error: %s\n", cudaGetErrorString(err));
    }
    printf("after kernel\n");
    
     // Wait for GPU to finish
     CUDA_CHECK(cudaDeviceSynchronize());
     
     printf("cuda_process_frame completed successfully\n");
     return 1;
 }