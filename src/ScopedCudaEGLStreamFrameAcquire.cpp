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
            r = cuGraphicsResourceGetMappedEglFrame(&m_frame, m_resource, 0, 0);

            if (r == CUDA_SUCCESS)
            {
                printf("Constructor finished!\n");
            }
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

        // size_t numElem = cudaEgl->planeDesc[0].pitch * cudaEgl->planeDesc[0].height;

        size_t width = m_frame.width;
        size_t height = m_frame.height;
        size_t pith = m_frame.pitch;
        size_t planeCount = m_frame.planeCount;

        printf("===\n");
        printf("width: %u\n", width);
        printf("height: %u\n", height);
        printf("pith: %u\n", pith);
        printf("planeCount: %u\n", planeCount);

        size_t ARRAY_SIZE = width * height;
        size_t ARRAY_BYTES = sizeof(uchar) * ARRAY_SIZE;

        CUarray cuArray = m_frame.frame.pArray[0];
        // uchar *d_ptr = (uchar *)m_frame.frame.pPitch[0];
        uchar *h_ptr = (uchar *)malloc(ARRAY_BYTES);

        printf("%p\n", &cuArray);
        // printf("%p\n", d_ptr);
        printf("%p\n", h_ptr);

        // cudaError_t r = cudaMemcpy((void *)h_ptr, (void *)d_ptr, ARRAY_BYTES, cudaMemcpyDeviceToHost);

        // const char* str = cudaGetErrorString(r);

        // printf("%s\n", str);

        // 640x480
        // 640/2 = 320 | 290 320 350
        // 480/2 = 240 | 210 240 270

        // size_t yStart = 210;
        // size_t xStart = 290;

        // for (size_t y = 0; y < 60; y++)
        // {
        //     for (size_t x = 0; x < 60; x++)
        //     {
        //         size_t posY = (yStart + y) * width;
        //         size_t posX = (xStart + x);
        //         size_t idx = posY + posX;
        //         printf("%u ", d_ptr[idx]);
        //     }
        //     printf("\n");
        // }

        // free(d_ptr);

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