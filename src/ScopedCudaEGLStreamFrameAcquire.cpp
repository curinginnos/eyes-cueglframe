#include "ScopedCudaEGLStreamFrameAcquire.hpp"

using namespace Argus;
using namespace cv;

namespace ArgusSamples
{
    ScopedCudaEGLStreamFrameAcquire::ScopedCudaEGLStreamFrameAcquire(CUeglStreamConnection &connection)
        : m_connection(connection), m_stream(NULL), m_resource(0)
    {
        CUresult r = cuEGLStreamConsumerAcquireFrame(&m_connection, &m_resource, &m_stream, -1);
        if (r == CUDA_SUCCESS)
        {
            cuGraphicsResourceGetMappedEglFrame(&m_frame, m_resource, 0, 0);
        }
    }

    ScopedCudaEGLStreamFrameAcquire::~ScopedCudaEGLStreamFrameAcquire()
    {
        if (m_resource)
        {
            cuEGLStreamConsumerReleaseFrame(&m_connection, m_resource, &m_stream);
        }
    }

    bool ScopedCudaEGLStreamFrameAcquire::hasValidFrame() const
    {
        return m_resource && checkFormat();
    }

    bool ScopedCudaEGLStreamFrameAcquire::generateHistogram(unsigned int histogramData[HISTOGRAM_BINS],
                                                            float *time)
    {
        if (!hasValidFrame() || !histogramData || !time)
            ORIGINATE_ERROR("Invalid state or output parameters");

        // Create surface from luminance channel.
        CUDA_RESOURCE_DESC cudaResourceDesc;
        memset(&cudaResourceDesc, 0, sizeof(cudaResourceDesc));
        cudaResourceDesc.resType = CU_RESOURCE_TYPE_ARRAY;
        cudaResourceDesc.res.array.hArray = m_frame.frame.pArray[0];
        CUsurfObject cudaSurfObj = 0;
        CUresult cuResult = cuSurfObjectCreate(&cudaSurfObj, &cudaResourceDesc);
        if (cuResult != CUDA_SUCCESS)
        {
            ORIGINATE_ERROR("Unable to create the surface object (CUresult %s)",
                            getCudaErrorString(cuResult));
        }

        // Generated the histogram.
        // *time += histogram(cudaSurfObj, m_frame.width, m_frame.height, histogramData);

        // Destroy surface.
        cuSurfObjectDestroy(cudaSurfObj);

        return true;
    }

    Size2D<uint32_t> ScopedCudaEGLStreamFrameAcquire::getSize() const
    {
        if (hasValidFrame())
            return Size2D<uint32_t>(m_frame.width, m_frame.height);
        return Size2D<uint32_t>(0, 0);
    }

    bool ScopedCudaEGLStreamFrameAcquire::checkFormat() const
    {
        if (!isCudaFormatYUV(m_frame.eglColorFormat))
        {
            ORIGINATE_ERROR("Only YUV color formats are supported");
        }
        if (m_frame.cuFormat != CU_AD_FORMAT_UNSIGNED_INT8)
        {
            ORIGINATE_ERROR("Only 8-bit unsigned int formats are supported");
        }
        return true;
    }
}