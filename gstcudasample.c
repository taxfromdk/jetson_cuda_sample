 #include <stdio.h>
 #include <stdlib.h>
 #include <string.h>
 #include <gst/gst.h>
 #include <gst/base/gstbasetransform.h>
 #include <gst/video/video.h>
 #include <gst/allocators/gstdmabuf.h>
 #include <cuda_runtime.h>
 #include "cudahelper.h"
 #include "gstcudasample.h"
 
  
 /* Add missing NVIDIA headers */
 #include <nvbufsurface.h>  /* For NvBufSurface related functions and types */
 
 /* Define NVMM feature string if not already defined */
 #ifndef GST_CAPS_FEATURE_MEMORY_NVMM
 #define GST_CAPS_FEATURE_MEMORY_NVMM "memory:NVMM"
 #endif
 
 /* Define NVBUF memory types if not already defined */
 #ifndef NVBUF_MEM_CUDA
 #define NVBUF_MEM_CUDA 2
 #endif
 
 #ifndef NVBUF_MEM_SURFACE_ARRAY
 #define NVBUF_MEM_SURFACE_ARRAY 4
 #endif
 
 GST_DEBUG_CATEGORY_STATIC (gst_cuda_sample_debug);
 #define GST_CAT_DEFAULT gst_cuda_sample_debug
 
 enum
 {
   PROP_0,
   PROP_ENABLED,
   PROP_GPU_ID
 };
 
 /* Define static caps for the plugin */
 static GstStaticPadTemplate sink_factory = GST_STATIC_PAD_TEMPLATE ("sink",
     GST_PAD_SINK,
     GST_PAD_ALWAYS,
     GST_STATIC_CAPS ("video/x-raw(memory:NVMM), "
         "format = (string) NV12, "
         "width = (int) [ 1, MAX ], "
         "height = (int) [ 1, MAX ], "
         "framerate = (fraction) [ 0/1, MAX ]")
     );
 
 static GstStaticPadTemplate src_factory = GST_STATIC_PAD_TEMPLATE ("src",
     GST_PAD_SRC,
     GST_PAD_ALWAYS,
     GST_STATIC_CAPS ("video/x-raw(memory:NVMM), "
         "format = (string) NV12, "
         "width = (int) [ 1, MAX ], "
         "height = (int) [ 1, MAX ], "
         "framerate = (fraction) [ 0/1, MAX ]")
     );
 
 #define gst_cuda_sample_parent_class parent_class
 G_DEFINE_TYPE (GstCudaSample, gst_cuda_sample, GST_TYPE_BASE_TRANSFORM);
 
 /* Function prototypes */
 static void gst_cuda_sample_set_property (GObject * object, guint prop_id, const GValue * value, GParamSpec * pspec);
 static void gst_cuda_sample_get_property (GObject * object, guint prop_id, GValue * value, GParamSpec * pspec);
 static GstFlowReturn gst_cuda_sample_transform_ip (GstBaseTransform * base, GstBuffer * buf);
 static gboolean gst_cuda_sample_set_caps (GstBaseTransform * base, GstCaps * incaps, GstCaps * outcaps);
 static gboolean gst_cuda_sample_start (GstBaseTransform * base);
 static gboolean gst_cuda_sample_stop (GstBaseTransform * base);
 
 /* Initialize the cuda_sample's class */
 static void gst_cuda_sample_class_init (GstCudaSampleClass * klass)
 {
   GObjectClass *gobject_class = (GObjectClass *) klass;
   GstElementClass *gstelement_class = (GstElementClass *) klass;
   GstBaseTransformClass *basetransform_class = (GstBaseTransformClass *) klass;
 
   gobject_class->set_property = gst_cuda_sample_set_property;
   gobject_class->get_property = gst_cuda_sample_get_property;
 
   g_object_class_install_property (gobject_class, PROP_ENABLED,
       g_param_spec_boolean ("enabled", "Enabled", "Enable processing",
           TRUE, G_PARAM_READWRITE));
           
   g_object_class_install_property (gobject_class, PROP_GPU_ID,
       g_param_spec_int ("gpu-id", "GPU ID", "CUDA GPU device ID to use",
           0, G_MAXINT, 0, G_PARAM_READWRITE));
 
   basetransform_class->transform_ip = GST_DEBUG_FUNCPTR (gst_cuda_sample_transform_ip);
   basetransform_class->set_caps = GST_DEBUG_FUNCPTR (gst_cuda_sample_set_caps);
   basetransform_class->start = GST_DEBUG_FUNCPTR (gst_cuda_sample_start);
   basetransform_class->stop = GST_DEBUG_FUNCPTR (gst_cuda_sample_stop);
   basetransform_class->passthrough_on_same_caps = TRUE;
 
   /* Add pad templates */
   gst_element_class_add_pad_template (gstelement_class,
       gst_static_pad_template_get (&src_factory));
   gst_element_class_add_pad_template (gstelement_class,
       gst_static_pad_template_get (&sink_factory));
 
   gst_element_class_set_static_metadata (gstelement_class,
       "CUDA Sample", "Filter/Effect/Video",
       "Manipulate video with CUDA",
       "Your Name <your.email@example.com>");
 
   GST_DEBUG_CATEGORY_INIT (gst_cuda_sample_debug, "cudasample", 0,
       "CUDA Sample");
 }
 
 /* Initialize the new element */
static void gst_cuda_sample_init (GstCudaSample * overlay)
{
    printf("Initializing variables\n");
    overlay->process_enabled = TRUE;
    overlay->gpu_id = 0;
    overlay->frame_num = 0;  /* Initialize frame counter */
}

static void
gst_cuda_sample_finalize (GObject * object)
{
    printf("Deallocating\n");
    GstCudaSample *overlay = GST_CUDA_SAMPLE (object);

    if (overlay->temp_surface_initialized) 
    {
        NvBufSurfaceDestroy(overlay->temp_surface);
        cudaFree(overlay->unified_sharpness_buffer);
        overlay->temp_surface_initialized = false;
    }
    G_OBJECT_CLASS (parent_class)->finalize (object);
}


 
 static void
 gst_cuda_sample_set_property (GObject * object, guint prop_id,
     const GValue * value, GParamSpec * pspec)
 {
   GstCudaSample *overlay = GST_cuda_sample (object);
 
   switch (prop_id) {
     case PROP_ENABLED:
       overlay->process_enabled = g_value_get_boolean (value);
       break;
     case PROP_GPU_ID:
       overlay->gpu_id = g_value_get_int (value);
       break;
     default:
       G_OBJECT_WARN_INVALID_PROPERTY_ID (object, prop_id, pspec);
       break;
   }
 }
 
 static void
 gst_cuda_sample_get_property (GObject * object, guint prop_id,
     GValue * value, GParamSpec * pspec)
 {
   GstCudaSample *overlay = GST_cuda_sample (object);
 
   switch (prop_id) {
     case PROP_ENABLED:
       g_value_set_boolean (value, overlay->process_enabled);
       break;
     case PROP_GPU_ID:
       g_value_set_int (value, overlay->gpu_id);
       break;
     default:
       G_OBJECT_WARN_INVALID_PROPERTY_ID (object, prop_id, pspec);
       break;
   }
 }
 
 static gboolean
 gst_cuda_sample_set_caps (GstBaseTransform * base, GstCaps * incaps,
     GstCaps * outcaps)
 {
   GstCudaSample *overlay = GST_cuda_sample (base);
   GstStructure *structure;
   const gchar *format;
   gint width, height;
   gint fps_n, fps_d;
 
   GST_DEBUG_OBJECT (overlay, "Setting caps: %" GST_PTR_FORMAT, incaps);
 
   /* Check if we have NVMM memory */
   structure = gst_caps_get_structure (incaps, 0);
   GstCapsFeatures *features = gst_caps_get_features (incaps, 0);
   if (!gst_caps_features_contains (features, GST_CAPS_FEATURE_MEMORY_NVMM)) {
     GST_ERROR_OBJECT (overlay, "Input caps missing NVMM memory feature");
     return FALSE;
   }
 
   /* Verify we have the expected format */
   format = gst_structure_get_string (structure, "format");
   if (!format || g_strcmp0 (format, "NV12") != 0) {
     GST_ERROR_OBJECT (overlay, "Input caps has unsupported format: %s", format ? format : "NULL");
     return FALSE;
   }
 
   /* Check resolution */
   if (!gst_structure_get_int (structure, "width", &width) ||
       !gst_structure_get_int (structure, "height", &height) ||
       width <= 0 || height <= 0) {
     GST_ERROR_OBJECT (overlay, "Failed to get valid dimensions from caps");
     return FALSE;
   }
 
   /* Optionally check framerate */
   if (gst_structure_get_fraction (structure, "framerate", &fps_n, &fps_d)) {
     GST_DEBUG_OBJECT (overlay, "Framerate: %d/%d", fps_n, fps_d);
   }
 
   /* Parse the input caps to get video info */
   if (!gst_video_info_from_caps (&overlay->video_info, incaps)) {
     GST_ERROR_OBJECT (overlay, "Failed to parse caps into video info");
     return FALSE;
   }
 
   GST_DEBUG_OBJECT (overlay, "Configured for %dx%d NV12 video", width, height);
 
   /* We don't use outcaps here since we're in-place transform */
   (void)outcaps;
 
   return TRUE;
 }
 
 static gboolean
 gst_cuda_sample_start (GstBaseTransform * base)
 {
   GstCudaSample *overlay = GST_cuda_sample (base);
 
   GST_DEBUG_OBJECT (overlay, "Start");
   
 
   return TRUE;
 }
 
 static gboolean
 gst_cuda_sample_stop (GstBaseTransform * base)
 {
   GstCudaSample *overlay = GST_cuda_sample (base);
 
   GST_DEBUG_OBJECT (overlay, "Stop");
   
   return TRUE;
 }
 

void handleSurface(GstCudaSample *filter, NvBufSurface *surface)
{
    NvBufSurfaceParams* src_params = &surface->surfaceList[0];
    if (!filter->temp_surface_initialized) 
    {

        NvBufSurfaceCreateParams create_params = {0};
        create_params.gpuId = filter->gpu_id;
        create_params.width = src_params->width;
        create_params.height = src_params->height;
        create_params.colorFormat = NVBUF_COLOR_FORMAT_NV12;
        create_params.layout = NVBUF_LAYOUT_PITCH;
        create_params.memType = NVBUF_MEM_DEFAULT; // CPU accessible

        if (NvBufSurfaceCreate(&filter->temp_surface, 1, &create_params) != 0) 
        {
            GST_ERROR("Failed to create temp surface");
            return;
        }
        cudaMallocManaged(&filter->unified_sharpness_buffer, src_params->width*src_params->height, cudaMemAttachGlobal);
        printf("\n\n\nunified dharpness buffer %p\n", filter->unified_sharpness_buffer);
        filter->temp_surface_initialized = true;
    }


    // Map NVMM buffer to CPU
    // Corrected buffer mappings
    if (NvBufSurfaceMap(surface, -1, -1, NVBUF_MAP_READ_WRITE) != 0 || NvBufSurfaceMap(filter->temp_surface, -1, -1, NVBUF_MAP_READ_WRITE) != 0) 
    {
        GST_ERROR("Surface mapping failed");
        return;
    }

    NvBufSurfaceSyncForCpu(surface, -1, -1);
    NvBufSurfaceSyncForCpu(filter->temp_surface, -1, -1);


    // Copy NVMM->CPU buffer
    cudaMemcpy2DAsync(
        filter->temp_surface->surfaceList[0].mappedAddr.addr[0],
        filter->temp_surface->surfaceList[0].planeParams.pitch[0],
        surface->surfaceList[0].mappedAddr.addr[0],
        surface->surfaceList[0].planeParams.pitch[0],
        src_params->width,
        src_params->height,
        cudaMemcpyHostToHost,
        filter->cuda_stream
    );

    // Ensure copy is done
    cudaStreamSynchronize(0);

    // Now run your CUDA kernel (temp_surface must be mapped with cudaHostRegister for CUDA kernel directly)
    cudaHostRegister(filter->temp_surface->surfaceList[0].mappedAddr.addr[0],
                        src_params->height * filter->temp_surface->surfaceList[0].planeParams.pitch[0],
                        cudaHostRegisterMapped);

    unsigned char* y_plane_dev;
    cudaHostGetDevicePointer(&y_plane_dev, filter->temp_surface->surfaceList[0].mappedAddr.addr[0], 0);

    cuda_process_frame(y_plane_dev,
        src_params->width,
        src_params->height,
        filter->temp_surface->surfaceList[0].planeParams.pitch[0]);

    cudaStreamSynchronize(0);
    cudaHostUnregister(filter->temp_surface->surfaceList[0].mappedAddr.addr[0]);


    // Copy CPU buffer back to NVMM buffer
    cudaMemcpy2DAsync(
        surface->surfaceList[0].mappedAddr.addr[0],
        surface->surfaceList[0].planeParams.pitch[0],
        filter->temp_surface->surfaceList[0].mappedAddr.addr[0],
        filter->temp_surface->surfaceList[0].planeParams.pitch[0],
        src_params->width,
        src_params->height,
        cudaMemcpyHostToHost,
        NULL
    );

    cudaStreamSynchronize(0);

    NvBufSurfaceSyncForDevice(surface, -1, -1);
    NvBufSurfaceUnMap(filter->temp_surface, -1, -1);
    
}



 static GstFlowReturn gst_cuda_sample_transform_ip (GstBaseTransform * trans, GstBuffer * buf)
 {
    GstCudaSample *filter = GST_cuda_sample (trans);
    GstFlowReturn ret = GST_FLOW_OK;
    GstMapInfo in_map_info;
    NvBufSurface *surface = NULL;

    filter->frame_num++;

    GST_DEBUG_OBJECT(filter, "Processing Frame %lu", filter->frame_num);

    /* Skip processing if not enabled */
    if (!filter->process_enabled) {
        GST_DEBUG_OBJECT(filter, "Processing disabled, skipping frame");
        return GST_FLOW_OK;
    }

    // Validate buffer
    if (!buf || gst_buffer_get_size(buf) == 0) {
        GST_WARNING_OBJECT(filter, "Empty buffer received, skipping");
        return GST_FLOW_OK;
    }

    // Map the buffer for read/write access
    memset(&in_map_info, 0, sizeof(in_map_info));
    if (!gst_buffer_map(buf, &in_map_info, GST_MAP_READ | GST_MAP_WRITE)) {
        GST_ERROR_OBJECT(filter, "Failed to map gst buffer");
        return GST_FLOW_ERROR;
    }

    // Get the NvBufSurface from the mapped buffer
    surface = (NvBufSurface *)in_map_info.data;
    if (!surface) {
        GST_ERROR_OBJECT(filter, "Failed to get NvBufSurface from mapped buffer");
        gst_buffer_unmap(buf, &in_map_info);
        return GST_FLOW_ERROR;
    }

    handleSurface(filter, surface);

    // Unmap the surface
    NvBufSurfaceUnMap(surface, -1, -1);

    // Unmap the buffer
    gst_buffer_unmap(buf, &in_map_info);

    return ret;
}
 
 /* Plugin initialization function */
 static gboolean
 plugin_init (GstPlugin * plugin)
 {
   return gst_element_register (plugin, "cudasample", GST_RANK_NONE,
       GST_TYPE_CUDA_SAMPLE);
 }
 
 /* Plugin details for registration */
 #define VERSION "1.0"
 #define PACKAGE "cudasample"
 #define ORIGIN "example.com"
 GST_PLUGIN_DEFINE (GST_VERSION_MAJOR, GST_VERSION_MINOR,
     cudasample, "CUDA Sample plugin", plugin_init, VERSION, "LGPL", PACKAGE, ORIGIN)