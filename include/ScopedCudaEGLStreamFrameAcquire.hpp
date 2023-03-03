#pragma once

#include <Argus/Argus.h>
#include <iostream>
#include <cuda.h>
#include <cudaEGL.h>

#include "opencv2/core/cuda.hpp"

namespace ArgusSamples
{
    // Debug print macros.
    #define PRODUCER_PRINT(...) printf("PRODUCER: " __VA_ARGS__)
    #define CONSUMER_PRINT(...) printf("CONSUMER: " __VA_ARGS__)


    /**
     * Utility class to acquire and process an EGLStream frame from a CUDA
     * consumer as long as the object is in scope.
     */
    class ScopedCudaEGLStreamFrameAcquire
    {
    public:
        /**
         * Constructor blocks until a frame is acquired or an error occurs (ie. stream is
         * disconnected). Caller should check which condition was satisfied using hasValidFrame().
         */
        ScopedCudaEGLStreamFrameAcquire(CUeglStreamConnection &connection);

        /**
         * Destructor releases frame back to EGLStream.
         */
        ~ScopedCudaEGLStreamFrameAcquire();

        /**
         * Returns true if a frame was acquired (and is compatible with this consumer).
         */
        bool hasValidFrame() const;

        cv::cuda::GpuMat getGpuMat() const; 

        /**
         * Returns the size (resolution) of the frame.
         */
        Argus::Size2D<uint32_t> getSize() const;

    private:
        /**
         * Returns whether or not the frame format is supported.
         */
        bool checkFormat() const;
        
        CUeglStreamConnection &m_connection;
        CUstream m_stream;
        CUgraphicsResource m_resource;
        CUeglFrame m_frame;
    };

}