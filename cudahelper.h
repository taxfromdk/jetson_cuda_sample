 #ifndef __CUDA_HELPER_H__
 #define __CUDA_HELPER_H__
 
#include <stdint.h>

 #ifdef __cplusplus
 extern "C" {
 #endif
 
int cuda_compute_sharpness(void* y_ptr, void* s_ptr, uint16_t width, uint16_t height, uint16_t pitch);
int cuda_process_frame(void* y_plane, uint16_t width, uint16_t height, uint16_t pitch, float timestamp_sec);
 
 #ifdef __cplusplus
 }
 #endif
 
 #endif