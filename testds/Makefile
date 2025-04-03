CC = gcc
CXX = g++
NVCC = nvcc

# Add -fPIC to C++ compilation flags
CXXFLAGS = -fPIC

DEEPSTREAM_PATH = /opt/nvidia/deepstream/deepstream-7.1

# Core DeepStream includes
NVMM_CFLAGS = -I. \
              -I$(DEEPSTREAM_PATH)/sources/includes \
              -I/usr/src/jetson_multimedia_api/include

# DeepStream libraries - adding NvBufSurfTransform
NVMM_LIBS = -L$(DEEPSTREAM_PATH)/lib -Wl,-rpath,$(DEEPSTREAM_PATH)/lib \
            -lnvbufsurface -lnvbufsurftransform -lnvdsgst_meta -lnvds_meta \
            -lnvdsbufferpool -lnvds_utils

# CUDA settings
CUDA_CFLAGS = -I/usr/local/cuda/include
CUDA_LIBS = -L/usr/local/cuda/lib64 -lcudart -lcuda

# GStreamer settings
GST_CFLAGS = $(shell pkg-config --cflags gstreamer-1.0 gstreamer-video-1.0 gstreamer-base-1.0)
GST_LIBS = $(shell pkg-config --libs gstreamer-1.0 gstreamer-video-1.0 gstreamer-base-1.0)

# Explicitly add the path to gstbasetransform.h
GST_CFLAGS += -I/usr/include/gstreamer-1.0/gst/base

test: libmycudaproc.so
	GST_DEBUG=1,nvdsvideotemplate:5 GST_DEBUG=LD_LIBRARY_PATH=$(DEEPSTREAM_PATH)/lib:$$LD_LIBRARY_PATH gst-launch-1.0 nvarguscamerasrc sensor-id=1 ! \
		"video/x-raw(memory:NVMM),format=NV12,width=3840,height=2160,framerate=30/1" ! \
		nvvidconv bl-output=false ! \
		nvdsvideotemplate customlib-name=./libmycudaproc.so ! \
		nvv4l2h265enc iframeinterval=90 maxperf-enable=true insert-sps-pps=true preset-level=1 insert-vui=true insert-aud=true bitrate=8000000 ! \
		h265parse config-interval=-1 ! \
		queue ! \
		mpegtsmux ! \
		tcpserversink host="0.0.0.0" port=9002

libmycudaproc.so: mycudalib.cpp cudahelper.o
	$(CXX) $(CXXFLAGS) -shared -o $@ $^ \
	$(NVMM_CFLAGS) $(NVMM_LIBS) \
	$(CUDA_CFLAGS) $(CUDA_LIBS) \
	$(GST_CFLAGS) $(GST_LIBS)

%.o: %.cu
	$(NVCC) -ccbin $(CXX) --compiler-options '-fPIC' -I. -c $< -o $@
	
clean: 
	rm -rf libmycudaproc.so cudahelper.o

.PHONY: test clean