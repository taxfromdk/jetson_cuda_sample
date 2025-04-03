 #ifndef __CUDA_HELPER_H__
 #define __CUDA_HELPER_H__
 
#include <stdint.h>

 #ifdef __cplusplus
 extern "C" {
 #endif
 
 int cuda_process_frame(int gpuId, void* y_plane, uint16_t width, uint16_t height, uint16_t pitch);
 
 #ifdef __cplusplus
 }
 #endif
 
 #endif