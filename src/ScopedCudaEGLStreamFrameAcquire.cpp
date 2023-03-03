#include "ScopedCudaEGLStreamFrameAcquire.hpp"

#include "opencv2/core.hpp"
#include "opencv2/core/cuda.hpp"
#include "opencv2/cudaarithm.hpp"
#include "opencv2/cudafilters.hpp"
#include "opencv2/cudaimgproc.hpp"
#include "opencv2/highgui.hpp"

// isCudaFormatYUV
#include "CUDAHelper.h"


#include <npp.h>
#include <nppi.h>
#include <nppdefs.h>
#include <nppi_color_conversion.h>

#include "cuUtils.hpp"
#include <fstream>

#include "Error.h"


using namespace Argus;
using namespace cv;

#define CUDA_TO_NPP 0

namespace ArgusSamples
{
    void dumpArrayFile(uchar *arr, int len, const char *name, int idx)
    {
        char filename[50];

        sprintf(filename, "%s_%d.txt", name, idx);

        std::ofstream ofs(filename);
        if (ofs.is_open())
        {
            for (size_t i = 0; i < len; i++)
            {
                ofs << ((int)arr[i]) << ",";
            }
            ofs.close();
        }

        printf("[MSG] %s saved!\n", filename);
    }

    void dumpImageFile(uchar *arr, const char *name, int idx, int width, int height)
    {
        char filename[50];

        sprintf(filename, "%s_%d.png", name, idx);

        cv::Mat frame;
        frame.create(height, width, CV_8U);
        frame.data = arr;

        cv::imwrite(filename, frame);
    }

    void checkError(cudaError_t err, const char *func)
    {
        if (err)
            printf("[ERR] %s\n", cudaGetErrorString(err));
        else
            printf("[MSG] %s succesful!\n", func);
    }

    void checkNppError(NppStatus err, const char *func)
    {
    }

    void checkNppStatus(NppStatus err, const char *func)
    {
        if (err)
            printf("[ERR] Failed!\n");
        else
            printf("[MSG] %s succesful\n", func);
    }

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

    bool ScopedCudaEGLStreamFrameAcquire::cvtNV12toBGR() const
    {
        CUarray cuY = m_frame.frame.pArray[0];
        CUarray cuCrCb = m_frame.frame.pArray[1];
        
        const size_t HEIGHT = m_frame.height;
        const size_t WIDTH = m_frame.width;
        const size_t HEIGHT_HALF = HEIGHT / 2;
        const size_t WIDTH_HALF = WIDTH / 2;
        const size_t HEIGHT_HALF_HALF = HEIGHT / 4;
        const size_t WIDTH_HALF_HALF = WIDTH / 4;
        const size_t CHANNEL = 3;

        cudaError_t err;
        NppStatus nppErr;

        uchar *d_Y;
        uchar *d_CrCb;
        cudaMalloc(&d_Y, sizeof(uchar) * WIDTH * HEIGHT);
        cudaMalloc(&d_CrCb, sizeof(uchar) * WIDTH * HEIGHT_HALF);

        // Retrieve Y and CbCr palnes
        err = cudaMemcpy2DFromArray(d_Y,
                                    WIDTH * sizeof(uchar),
                                    (cudaArray_t)cuY,
                                    0,
                                    0,
                                    WIDTH * sizeof(uchar),
                                    HEIGHT,
                                    cudaMemcpyDeviceToDevice);

        checkError(err, "cudaMemcpy2DFromArray - Y");

        err = cudaMemcpy2DFromArray(d_CrCb,
                                    WIDTH * sizeof(uchar),
                                    (cudaArray_t)cuCrCb,
                                    0,
                                    0,
                                    WIDTH * sizeof(uchar),
                                    HEIGHT_HALF,
                                    cudaMemcpyDeviceToDevice);

        checkError(err, "cudaMemcpy2DFromArray - CrCb");

        Npp8u *const pSrc[2] = {d_Y, d_CrCb};
        int rSrcStep = WIDTH * sizeof(uchar);
        int nDstStep = WIDTH * 3 * sizeof(uchar);
        NppiSize oSizeROI = {WIDTH, HEIGHT};

        uchar *d_bgr;
        cudaMalloc(&d_bgr, sizeof(uchar) * WIDTH * HEIGHT * 3);

        nppErr = nppiNV12ToBGR_8u_P2C3R(pSrc, rSrcStep, d_bgr, nDstStep, oSizeROI);            

        if(nppErr)
        {
            printf("%d\n", nppErr);
        }

        cv::cuda::GpuMat gpuMatBGR;
        gpuMatBGR.create(HEIGHT, WIDTH, CV_8UC3);
        gpuMatBGR.data = d_bgr;
        gpuMatBGR.step = WIDTH * 3 * sizeof(uchar);

        cv::Mat cpuMatBGR;
        cpuMatBGR.create(HEIGHT, WIDTH, CV_8UC3);
        cpuMatBGR.step = WIDTH * 3 * sizeof(uchar);

        gpuMatBGR.download(cpuMatBGR);

        cv::imshow("img", cpuMatBGR);
        cv::pollKey();

        cudaFree(d_Y);
        cudaFree(d_CrCb);
        
        return true;
    }

    bool ScopedCudaEGLStreamFrameAcquire::generateHistogram()
    {
        if (!hasValidFrame())
            ORIGINATE_ERROR("Invalid state or output parameters");

        cvtNV12toBGR();

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