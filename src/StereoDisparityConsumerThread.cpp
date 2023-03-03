#include "StereoDisparityConsumerThread.hpp"
#include "ScopedCudaEGLStreamFrameAcquire.hpp"

#include "Error.h"

using namespace Argus;

namespace ArgusSamples
{

    bool StereoDisparityConsumerThread::threadInitialize()
    {
        // Create CUDA and connect egl streams.
        PROPAGATE_ERROR(initCUDA(&m_cudaContext));

        CONSUMER_PRINT("Connecting CUDA consumer to left stream\n");
        CUresult cuResult = cuEGLStreamConsumerConnect(&m_cuStreamLeft, m_leftStream->getEGLStream());
        if (cuResult != CUDA_SUCCESS)
        {
            ORIGINATE_ERROR("Unable to connect CUDA to EGLStream as a consumer (CUresult %s)",
                            getCudaErrorString(cuResult));
        }

        CONSUMER_PRINT("Connecting CUDA consumer to right stream\n");
        cuResult = cuEGLStreamConsumerConnect(&m_cuStreamRight, m_rightStream->getEGLStream());
        if (cuResult != CUDA_SUCCESS)
        {
            ORIGINATE_ERROR("Unable to connect CUDA to EGLStream as a consumer (CUresult %s)",
                            getCudaErrorString(cuResult));
        }
        return true;
    }

    bool StereoDisparityConsumerThread::threadExecute()
    {
        CONSUMER_PRINT("Waiting for Argus producer to connect to left stream.\n");
        m_leftStream->waitUntilConnected();

        CONSUMER_PRINT("Waiting for Argus producer to connect to right stream.\n");
        m_rightStream->waitUntilConnected();

        CONSUMER_PRINT("Streams connected, processing frames.\n");
        while (true)
        {
            CONSUMER_PRINT("RETRIEVING FRAME\n");

            EGLint streamState = EGL_STREAM_STATE_CONNECTING_KHR;

            // Check both the streams and proceed only if they are not in DISCONNECTED state.
            if (!eglQueryStreamKHR(
                    m_leftStream->getEGLDisplay(),
                    m_leftStream->getEGLStream(),
                    EGL_STREAM_STATE_KHR,
                    &streamState) ||
                (streamState == EGL_STREAM_STATE_DISCONNECTED_KHR))
            {
                CONSUMER_PRINT("left : EGL_STREAM_STATE_DISCONNECTED_KHR received\n");
                break;
            }

            if (!eglQueryStreamKHR(
                    m_rightStream->getEGLDisplay(),
                    m_rightStream->getEGLStream(),
                    EGL_STREAM_STATE_KHR,
                    &streamState) ||
                (streamState == EGL_STREAM_STATE_DISCONNECTED_KHR))
            {
                CONSUMER_PRINT("right : EGL_STREAM_STATE_DISCONNECTED_KHR received\n");
                break;
            }

            ScopedCudaEGLStreamFrameAcquire left(m_cuStreamLeft);
            ScopedCudaEGLStreamFrameAcquire right(m_cuStreamRight);

            if(!left.cvtNV12toBGR())
                break;

            CONSUMER_PRINT("RETRIEVING FRAME FINISHED\n");
        }
        CONSUMER_PRINT("No more frames. Cleaning up.\n");

        PROPAGATE_ERROR(requestShutdown());

        return true;
    }

    bool StereoDisparityConsumerThread::threadShutdown()
    {
        // Disconnect from the streams.
        cuEGLStreamConsumerDisconnect(&m_cuStreamLeft);
        cuEGLStreamConsumerDisconnect(&m_cuStreamRight);

        PROPAGATE_ERROR(cleanupCUDA(&m_cudaContext));

        CONSUMER_PRINT("Done.\n");
        return true;
    }
}