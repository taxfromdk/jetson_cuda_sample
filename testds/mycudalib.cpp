#include <gst/gst.h>
#include <gst/base/gstbasetransform.h>
#include <cuda_runtime.h>

// Include DeepStream headers
#include "/opt/nvidia/deepstream/deepstream-7.1/sources/includes/nvbufsurface.h"
#include "/opt/nvidia/deepstream/deepstream-7.1/sources/includes/nvbufsurftransform.h"

// Custom library interfaces
#include "nvdscustomlib_interface.hpp"
#include "nvdscustomlib_factory.hpp"
#include "nvdscustomlib_base.hpp"

// CUDA kernel function declaration
extern "C" int cuda_process_frame(void* y_ptr, uint16_t width, uint16_t height, uint16_t pitch);
extern "C" int cuda_compute_sharpness(void* y_ptr, void* s_ptr, uint16_t width, uint16_t height, uint16_t pitch);



class MyCudaProcessor : public IDSCustomLibrary {
private:
    cudaStream_t cuda_stream;
    int gpu_id;

    NvBufSurfTransformConfigParams transform_config_params;
    NvBufSurfTransformParams transform_params;
    unsigned char* unified_sharpness_buffer;
    
    

public:
    MyCudaProcessor() : cuda_stream(nullptr), gpu_id(0), unified_sharpness_buffer(nullptr) {}

    ~MyCudaProcessor() override {
        cudaFree(unified_sharpness_buffer);
    }

    bool SetInitParams(DSCustom_CreateParams *params) override {
        cuda_stream = params->m_cudaStream;
        gpu_id = params->m_gpuId;
        return true;
    }

    BufferResult ProcessBuffer(GstBuffer *inbuf) override {
        GstMapInfo in_map_info;
    
        if (!gst_buffer_map(inbuf, &in_map_info, GST_MAP_READWRITE)) {
            GST_ERROR("Failed to map input buffer");
            return BufferResult::Buffer_Error;
        }
    
        NvBufSurface *surface = reinterpret_cast<NvBufSurface *>(in_map_info.data);
        if (!surface) {
            GST_ERROR("Failed to retrieve surface from buffer");
            gst_buffer_unmap(inbuf, &in_map_info);
            return BufferResult::Buffer_Error;
        }
    
        NvBufSurfaceParams* src_params = &surface->surfaceList[0];
    
        if( !unified_sharpness_buffer)
        {
            cudaMallocManaged(&unified_sharpness_buffer, 3840*2160);
            printf("\n\n\nunified dharpness buffer %p\n", unified_sharpness_buffer);
        }
        
        // Map NVMM buffer to CPU
        // Corrected buffer mappings
        if (NvBufSurfaceMap(surface, -1, -1, NVBUF_MAP_READ_WRITE) != 0) 
        {
            GST_ERROR("Surface mapping failed");
            gst_buffer_unmap(inbuf, &in_map_info);
            return BufferResult::Buffer_Error;
        }

        NvBufSurfaceSyncForCpu(surface, -1, -1);
        cudaHostRegister(surface->surfaceList[0].mappedAddr.addr[0],
                         src_params->height * src_params->planeParams.pitch[0],
                         cudaHostRegisterMapped);    
        unsigned char* y_plane_dev;
        cudaHostGetDevicePointer(&y_plane_dev, surface->surfaceList[0].mappedAddr.addr[0], 0);
    
        //************************************************
        //*   Sharpness
        //************************************************
        cuda_compute_sharpness(y_plane_dev, unified_sharpness_buffer, src_params->width, src_params->height, src_params->planeParams.pitch[0]);
        cudaStreamSynchronize(cuda_stream);
        
        uint64_t acc = 0; 
        for(int x = 0; x < 3840; x++)
        {
            for(int y = 0; y < 2160; y++)
            {
                acc += unified_sharpness_buffer[x+y*3840];
            }    
        }
        printf("Sharpness is %f\n", acc / (3840.0*2160));
        cudaStreamSynchronize(cuda_stream);

        //************************************************
        //*   Draw
        //************************************************
        cuda_process_frame(y_plane_dev, src_params->width, src_params->height, src_params->planeParams.pitch[0]);
    
        cudaStreamSynchronize(cuda_stream);
        cudaHostUnregister(surface->surfaceList[0].mappedAddr.addr[0]);
    
        NvBufSurfaceSyncForDevice(surface, -1, -1);
        NvBufSurfaceUnMap(surface, -1, -1);
        gst_buffer_unmap(inbuf, &in_map_info);
    
        return BufferResult::Buffer_Ok;
    }

    bool HandleEvent(GstEvent *) override { return true; }
    bool SetProperty(Property &) override { return true; }
    char* QueryProperties() override { return nullptr; }
    GstCaps* GetCompatibleCaps(GstPadDirection, GstCaps* caps, GstCaps*) override {
        return gst_caps_copy(caps);
    }
};

extern "C" IDSCustomLibrary *CreateCustomAlgoCtx(const gchar *, gpointer) {
    return new MyCudaProcessor();
}

extern "C" void DestroyCustomAlgoCtx(IDSCustomLibrary *ctx) {
    delete ctx;
}
