
 #ifndef __GST_CUDA_SAMPLE_H__
 #define __GST_CUDA_SAMPLE_H__
 
 #include <gst/gst.h>
 #include <gst/base/gstbasetransform.h>
 #include <gst/video/video.h>
 #include <gst/allocators/gstdmabuf.h>
 #include "/opt/nvidia/deepstream/deepstream-7.1/sources/includes/nvbufsurface.h"
 #include "/opt/nvidia/deepstream/deepstream-7.1/sources/includes/nvbufsurftransform.h"


 G_BEGIN_DECLS
 
 #define GST_TYPE_CUDA_SAMPLE (gst_cuda_sample_get_type())
 #define GST_cuda_sample(obj) (G_TYPE_CHECK_INSTANCE_CAST((obj),GST_TYPE_CUDA_SAMPLE,GstCudaSample))
 #define GST_cuda_sample_CLASS(klass) (G_TYPE_CHECK_CLASS_CAST((klass),GST_TYPE_CUDA_SAMPLE,GstCudaSampleClass))
 #define GST_IS_cuda_sample(obj) (G_TYPE_CHECK_INSTANCE_TYPE((obj),GST_TYPE_CUDA_SAMPLE))
 #define GST_IS_cuda_sample_CLASS(klass) (G_TYPE_CHECK_CLASS_TYPE((klass),GST_TYPE_CUDA_SAMPLE))
 
 typedef struct _GstCudaSample GstCudaSample;
 typedef struct _GstCudaSampleClass GstCudaSampleClass;
 
 struct _GstCudaSample
 {
   GstBaseTransform base_cuda_sample;
 
   /* Properties */
   gboolean process_enabled;
   gint gpu_id;
   
   /* Video format info */
   GstVideoInfo video_info;
   


   cudaStream_t cuda_stream;
   
   NvBufSurfTransformConfigParams transform_config_params;
   NvBufSurfTransformParams transform_params;
   NvBufSurface* temp_surface;
   bool temp_surface_initialized;

   void* unified_sharpness_buffer;
   




   /* Frame counter */
   gulong frame_num;
 };
 
 struct _GstCudaSampleClass
 {
   GstBaseTransformClass base_cuda_sample_class;
 };
 
 GType gst_cuda_sample_get_type (void);
 
 G_END_DECLS
 
 #endif /* __GST_CUDA_SAMPLE_H__ */