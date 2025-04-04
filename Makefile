# Makefile for building GStreamer Cuda Sample for Jetson with JetPack 6.2

# Compiler and flags
CC = gcc
CXX = g++
NVCC = nvcc

# Get the pkg-config paths and includes for GStreamer
GST_CFLAGS = $(shell pkg-config --cflags gstreamer-1.0 gstreamer-video-1.0 gstreamer-base-1.0 gstreamer-allocators-1.0)
GST_LIBS = $(shell pkg-config --libs gstreamer-1.0 gstreamer-video-1.0 gstreamer-base-1.0 gstreamer-allocators-1.0)  -lcudart -lcuda


# NVIDIA DeepStream specific includes and libraries
DEEPSTREAM_PATH = /opt/nvidia/deepstream/deepstream-7.1
NVMM_CFLAGS = -I$(DEEPSTREAM_PATH)/sources/includes -I/usr/src/jetson_multimedia_api/include
NVMM_LIBS = -L$(DEEPSTREAM_PATH)/lib -L/usr/lib/aarch64-linux-gnu/nvidia -lnvbufsurface

# Additional includes and libraries for CUDA
CUDA_CFLAGS = -I/usr/local/cuda/include
CUDA_LIBS = -L/usr/local/cuda/lib64 -lcudart

# Combined flags
CFLAGS = -Wall -Wextra -fPIC -O2 -g $(GST_CFLAGS) $(NVMM_CFLAGS) $(CUDA_CFLAGS)
LDFLAGS = $(GST_LIBS) $(NVMM_LIBS) $(CUDA_LIBS)

# Target plugin library name
TARGET = libgstcudasample.so

# Source files
SRCS = gstcudasample.c
CUDA_SRCS = cudahelper.cu

# Object files
OBJS = $(SRCS:.c=.o)
CUDA_OBJS = $(CUDA_SRCS:.cu=.o)

# Default target
all: $(TARGET)

# Build the plugin - build CUDA objects first to avoid dependency issues
$(TARGET): $(CUDA_OBJS) $(OBJS)
	$(CXX) -shared -o $@ $^ $(LDFLAGS)

# Compile C sources
%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

# Compile CUDA sources - using minimal includes to avoid conflict with GStreamer
%.o: %.cu
	$(NVCC) -ccbin $(CXX) --compiler-options '-fPIC' -I. -c $< -o $@

# Clean built files
clean:
	rm -f $(OBJS) $(CUDA_OBJS) $(TARGET)

test: $(TARGET)
	GST_DEBUG=1,cudasample:5 \
	GST_PLUGIN_PATH=. \
	gst-launch-1.0 \
		nvarguscamerasrc sensor-id=1 \
		! "video/x-raw(memory:NVMM),format=NV12,width=3840,height=2160,framerate=30/1" \
		! nvvidconv bl-output=false \
		! cudasample \
		! nvv4l2h265enc iframeinterval=90 maxperf-enable=true insert-sps-pps=true preset-level=1 insert-vui=true insert-aud=true bitrate=8000000 \
		! h265parse config-interval=-1 \
		! queue \
		! mpegtsmux \
		! tcpserversink host="0.0.0.0" port=9002

.PHONY: all clean install uninstall test

#Test video from external computer
#ffplay -fflags nobuffer -flags low_delay -framedrop -strict experimental -probesize 32 -sync ext tcp://host_ip:9002