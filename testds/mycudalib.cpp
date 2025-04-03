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
extern "C" int cuda_process_frame(int gpu_id, void* y_ptr, uint16_t width, uint16_t height, uint16_t pitch);

class MyCudaProcessor : public IDSCustomLibrary {
private:
    cudaStream_t cuda_stream;
    int gpu_id;

    NvBufSurfTransformConfigParams transform_config_params;
    NvBufSurfTransformParams transform_params;
    NvBufSurface* temp_surface;
    bool temp_surface_initialized;

    bool initTempSurface(int width, int height) {
        cleanup();

        NvBufSurfaceCreateParams create_params = {0};
        create_params.gpuId = gpu_id;
        create_params.width = width;
        create_params.height = height;
        create_params.colorFormat = NVBUF_COLOR_FORMAT_NV12;
        create_params.layout = NVBUF_LAYOUT_PITCH;
        create_params.memType = NVBUF_MEM_DEFAULT;

        int ret = NvBufSurfaceCreate(&temp_surface, 1, &create_params);
        if (ret != 0) {
            GST_ERROR("Failed to create temp surface: %d", ret);
            return false;
        }

        transform_config_params.compute_mode = NvBufSurfTransformCompute_Default;
        NvBufSurfTransformSetSessionParams(&transform_config_params);

        temp_surface_initialized = true;
        return true;
    }

    void cleanup() {
        if (temp_surface_initialized) {
            NvBufSurfaceDestroy(temp_surface);
            temp_surface_initialized = false;
        }
    }

public:
    MyCudaProcessor() : cuda_stream(nullptr), gpu_id(0), temp_surface(nullptr), temp_surface_initialized(false) {}

    ~MyCudaProcessor() override {
        cleanup();
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
    
        if (!temp_surface_initialized || 
            temp_surface->surfaceList[0].width != src_params->width ||
            temp_surface->surfaceList[0].height != src_params->height) {
            cleanup();
    
            NvBufSurfaceCreateParams create_params = {0};
            create_params.gpuId = gpu_id;
            create_params.width = src_params->width;
            create_params.height = src_params->height;
            create_params.colorFormat = NVBUF_COLOR_FORMAT_NV12;
            create_params.layout = NVBUF_LAYOUT_PITCH;
            create_params.memType = NVBUF_MEM_DEFAULT; // CPU accessible
    
            if (NvBufSurfaceCreate(&temp_surface, 1, &create_params) != 0) {
                GST_ERROR("Failed to create temp surface");
                gst_buffer_unmap(inbuf, &in_map_info);
                return BufferResult::Buffer_Error;
            }
            temp_surface_initialized = true;
        }
    
        // Map NVMM buffer to CPU
        // Corrected buffer mappings
        if (NvBufSurfaceMap(surface, -1, -1, NVBUF_MAP_READ_WRITE) != 0 || NvBufSurfaceMap(temp_surface, -1, -1, NVBUF_MAP_READ_WRITE) != 0) 
        {
            GST_ERROR("Surface mapping failed");
            gst_buffer_unmap(inbuf, &in_map_info);
            return BufferResult::Buffer_Error;
        }

        NvBufSurfaceSyncForCpu(surface, -1, -1);
        NvBufSurfaceSyncForCpu(temp_surface, -1, -1);

    
        // Copy NVMM->CPU buffer
        cudaMemcpy2DAsync(
            temp_surface->surfaceList[0].mappedAddr.addr[0],
            temp_surface->surfaceList[0].planeParams.pitch[0],
            surface->surfaceList[0].mappedAddr.addr[0],
            surface->surfaceList[0].planeParams.pitch[0],
            src_params->width,
            src_params->height,
            cudaMemcpyHostToHost,
            cuda_stream
        );
    
        // Ensure copy is done
        cudaStreamSynchronize(cuda_stream);
    
        // Now run your CUDA kernel (temp_surface must be mapped with cudaHostRegister for CUDA kernel directly)
        cudaHostRegister(temp_surface->surfaceList[0].mappedAddr.addr[0],
                         src_params->height * temp_surface->surfaceList[0].planeParams.pitch[0],
                         cudaHostRegisterMapped);
    
        unsigned char* y_plane_dev;
        cudaHostGetDevicePointer(&y_plane_dev, temp_surface->surfaceList[0].mappedAddr.addr[0], 0);
    
        cuda_process_frame(gpu_id,
                           y_plane_dev,
                           src_params->width,
                           src_params->height,
                           temp_surface->surfaceList[0].planeParams.pitch[0]);
    
        cudaStreamSynchronize(cuda_stream);
        cudaHostUnregister(temp_surface->surfaceList[0].mappedAddr.addr[0]);
    
        // Copy CPU buffer back to NVMM buffer
        cudaMemcpy2DAsync(
            surface->surfaceList[0].mappedAddr.addr[0],
            surface->surfaceList[0].planeParams.pitch[0],
            temp_surface->surfaceList[0].mappedAddr.addr[0],
            temp_surface->surfaceList[0].planeParams.pitch[0],
            src_params->width,
            src_params->height,
            cudaMemcpyHostToHost,
            cuda_stream
        );
    
        cudaStreamSynchronize(cuda_stream);
    
        NvBufSurfaceSyncForDevice(surface, -1, -1);
        NvBufSurfaceUnMap(surface, -1, -1);
        NvBufSurfaceUnMap(temp_surface, -1, -1);
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
