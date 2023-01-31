#include "StereoDisparityConsumerThread.hpp"
#include "ScopedCudaEGLStreamFrameAcquire.hpp"

using namespace Argus;
using namespace cv;

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
        unsigned int histogramLeft[HISTOGRAM_BINS];
        unsigned int histogramRight[HISTOGRAM_BINS];
        while (true)
        {
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

            if (!left.hasValidFrame() || !right.hasValidFrame())
                break;

            // Calculate histograms.
            float time = 0.0f;
            if (left.generateHistogram(histogramLeft, &time))
            {
                // // Calculate KL distance.
                // float distance = 0.0f;
                // Size2D<uint32_t> size = right.getSize();
                // float dTime = computeKLDistance(histogramRight,
                //                                 histogramLeft,
                //                                 HISTOGRAM_BINS,
                //                                 size.width() * size.height(),
                //                                 &distance);
                // CONSUMER_PRINT("KL distance of %6.3f with %5.2f ms computing histograms and "
                //                "%5.2f ms spent computing distance\n",
                //                distance, time, dTime);
            }
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