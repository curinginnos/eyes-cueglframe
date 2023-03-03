#pragma once

#include <Argus/Argus.h>
#include <cudaEGL.h>

#include "CommonOptions.h"
#include "CUDAHelper.h"
#include "EGLGlobal.h"
#include "Error.h"

#include "CUDAHelper.h"

#include "opencv2/core.hpp"
#include "opencv2/core/cuda.hpp"
#include "opencv2/cudaarithm.hpp"
#include "opencv2/cudafilters.hpp"
#include "opencv2/cudaimgproc.hpp"
#include "opencv2/highgui.hpp"

#include <npp.h>
#include <nppi.h>
#include <nppdefs.h>
#include <nppi_color_conversion.h>

#include "../cudaHistogram/histogram.h"

#include <iostream>


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

        /**
         * Use CUDA to generate a histogram from the acquired frame.
         * @param[out] histogramData Output array for the histogram.
         * @param[out] time Time to generate histogram, in milliseconds.
         */
        bool generateHistogram(unsigned int histogramData[HISTOGRAM_BINS], float *time);

        /**
         * Returns the size (resolution) of the frame.
         */
        Argus::Size2D<uint32_t> getSize() const;

    private:
        /**
         * Returns whether or not the frame format is supported.
         */
        bool checkFormat() const;
        bool cvtYUV2BGR() const;
        bool cvtCUDAYUV2BGR() const; // not supported
        bool cvtNPPYUV2BGR() const; // not supported
        bool cvtNV12toBGR() const; // not supported
        size_t calcByte();

        CUeglStreamConnection &m_connection;
        CUstream m_stream;
        CUgraphicsResource m_resource;
        CUeglFrame m_frame;
    };

}