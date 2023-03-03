#include "ScopedCudaEGLStreamFrameAcquire.hpp"
#include "cuUtils.hpp"
#include <fstream>

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

        // for (size_t i = 0; i < len; i++)
        // {
        //     // ofs << arr[i] << ",";
        //     printf("%d ", arr[i]);
        // }
        // printf("\n");
    }

    int FILE_IDX = 0;

    void checkError(cudaError_t err, const char *func)
    {
        if (err)
            printf("%s\n", cudaGetErrorString(err));
        else
            printf("[MSG] %s succesful!\n", func);
    }

    ScopedCudaEGLStreamFrameAcquire::ScopedCudaEGLStreamFrameAcquire(CUeglStreamConnection &connection)
        : m_connection(connection), m_stream(NULL), m_resource(0)
    {
        CUresult r = cuEGLStreamConsumerAcquireFrame(&m_connection, &m_resource, &m_stream, -1);
        if (r == CUDA_SUCCESS)
        {

            // Wrong!

            // r = cuGraphicsEGLRegisterImage(&m_resource,
            //                                &m_frame,
            //                                CU_GRAPHICS_MAP_RESOURCE_FLAGS_NONE);

            // if (r == CUDA_SUCCESS)
            // {
            //     printf("Called succes!!\n");
            // } else {
            //     const char* msg;

            //     cuGetErrorString(r, &msg);

            //     printf("%s\n", msg);
            // }

            r = cuGraphicsResourceGetMappedEglFrame(&m_frame, m_resource, 0, 0);

            if (r == CUDA_SUCCESS)
            {

                // Wrong code!

                // CUarray *arr = m_frame.frame.pArray;
                // void *ptr = m_frame.frame.pPitch;

                // cudaPointerAttributes attr;

                // cudaError_t err = cudaPointerGetAttributes(&attr, arr);

                // const char *errStr = cudaGetErrorString(err);

                // printf("%s\n", errStr);

                // printf("%d\n", attr.memoryType);
                // printf("%d\n", attr.type);
                // printf("%d\n", attr.device);
                // printf("%p\n", attr.devicePointer);
                // printf("%p\n", attr.hostPointer);
                // printf("%d\n", attr.isManaged);

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

        CUarray cuY = m_frame.frame.pArray[0];
        CUarray cuCrCb = m_frame.frame.pArray[1];

        const size_t HEIGHT = m_frame.height;
        const size_t WIDTH = m_frame.width;
        const size_t HEIGHT_HALF = HEIGHT / 2;
        const size_t WIDTH_HALF = WIDTH / 2;

        const size_t CHANNEL = 4;

        const size_t N = HEIGHT * WIDTH;
        const size_t ARRAY_BYTES = N * sizeof(uchar);
        cudaError_t err;

        CUarray cuArray;
        cudaArray_t cuArray2;

#ifdef CUDA_TO_NPP

        /**
         * CUDA Array -> Linear (Device to Device)
         *
         * Linear -> Linear (Device to Host)
         */

        uchar *d_Y;
        uchar *d_CrCb;
        uchar *d_bgr;
        uchar *d_Cr;
        uchar *d_Cb;

        uchar *h_Y = (uchar *)malloc(sizeof(uchar) * WIDTH * HEIGHT);
        uchar *h_CrCb = (uchar *)malloc(sizeof(uchar) * WIDTH_HALF * HEIGHT_HALF);
        uchar *h_bgr = (uchar *)malloc(sizeof(uchar) * WIDTH * HEIGHT * CHANNEL);
        uchar *h_Cr = (uchar *)malloc(sizeof(uchar) * WIDTH_HALF * HEIGHT_HALF / 2);
        uchar *h_Cb = (uchar *)malloc(sizeof(uchar) * WIDTH_HALF * HEIGHT_HALF / 2);

        // Width, Height
        cudaMalloc(&d_Y, sizeof(uchar) * WIDTH * HEIGHT);
        // Width/2, Height/2
        cudaMalloc(&d_CrCb, sizeof(uchar) * WIDTH_HALF * HEIGHT_HALF);
        // Width, Height, 3 channels
        cudaMalloc(&d_bgr, sizeof(uchar) * WIDTH * HEIGHT * CHANNEL);

        // Width/2, Height/2, / 2
        cudaMalloc(&d_Cr, sizeof(uchar) * WIDTH_HALF * HEIGHT_HALF / 2);
        // Width/2, Height/2, / 2
        cudaMalloc(&d_Cb, sizeof(uchar) * WIDTH_HALF * HEIGHT_HALF / 2);

        uchar *d_CrCb_middle = &d_CrCb[(WIDTH_HALF * HEIGHT_HALF / 2)];

        err = cudaMemcpy2DFromArray(d_Y,
                                    WIDTH * sizeof(uchar),
                                    (cudaArray_t)cuY,
                                    0,
                                    0,
                                    WIDTH * sizeof(uchar),
                                    HEIGHT,
                                    cudaMemcpyDeviceToDevice);

        checkError(err, "cudaMemcpy2DFromArray");

        err = cudaMemcpy2DFromArray(d_CrCb,
                                    WIDTH_HALF * sizeof(uchar),
                                    (cudaArray_t)cuCrCb,
                                    0,
                                    0,
                                    WIDTH_HALF * sizeof(uchar),
                                    HEIGHT_HALF,
                                    cudaMemcpyDeviceToDevice);

        checkError(err, "cudaMemcpy2DFromArray");

        unravel(d_CrCb, d_Cb, d_Cr, WIDTH_HALF, HEIGHT_HALF);

        err = cudaMemcpy(d_CrCb, d_Cr, sizeof(uchar) * WIDTH_HALF * HEIGHT_HALF / 2, cudaMemcpyDeviceToDevice);
        checkError(err, "cudaMemcpy");

        err = cudaMemcpy(d_CrCb_middle, d_Cb, sizeof(uchar) * WIDTH_HALF * HEIGHT_HALF / 2, cudaMemcpyDeviceToDevice);
        checkError(err, "cudaMemcpy");

        // int rSrcStep[] = {WIDTH, WIDTH_HALF, WIDTH_HALF};
        // uchar *pYUV420[] = {d_Y, d_Cr, d_Cb};

        uchar *pSrc[] = {d_Y, d_CrCb};
        int rSrcStep[] = {WIDTH, WIDTH_HALF, WIDTH_HALF};

        NppiSize roi{
            .width = static_cast<int>(WIDTH),
            .height = static_cast<int>(HEIGHT),
        };

        // NppStatus npperr = nppiYUV420ToBGR_8u_P3C3R(pYUV420,
        //                                             rSrcStep,
        //                                             d_bgr,
        //                                             static_cast<int>(WIDTH) * CHANNEL,
        //                                             roi);

        // NppStatus npperr = nppiYUV420ToBGR_8u_P3C4R(pYUV420,
        //                                             rSrcStep,
        //                                             d_bgr,
        //                                             static_cast<int>(WIDTH) * CHANNEL,
        //                                             roi);

        // NppStatus npperr = nppiNV12ToBGR_8u_P2C3R(pSrc,
        //                                           WIDTH * sizeof(uchar),
        //                                           d_bgr,
        //                                           WIDTH * sizeof(uchar),
        //                                           roi);

        // NppStatus npperr = nppiNV21ToBGR_8u_P2C4R(pSrc,
        //                                           WIDTH * sizeof(uchar),
        //                                           d_bgr,
        //                                           WIDTH * sizeof(uchar),
        //                                           roi);

        


        // cv::cuda::GpuMat gpuMat;
        // gpuMat.create(HEIGHT, WIDTH, CV_8UC4);
        // gpuMat.data = d_bgr;
        // gpuMat.step = WIDTH * CHANNEL;

        // cv::Mat cpuMat;
        // gpuMat.download(cpuMat);

        // cv::Mat converted;
        // cv::cvtColor(cpuMat, converted, COLOR_BGRA2BGR);

        // cv::imshow("cpuMat", cpuMat);
        // cv::imshow("converted", converted);
        // cv::waitKey(1);

        // 요기
        // err = cudaMemcpy(d_CrCb, d_Cb, sizeof(uchar) * WIDTH_HALF * HEIGHT_HALF / 2, cudaMemcpyDeviceToDevice);
        // checkError(err, "cudaMemcpy");
        // err = cudaMemcpy(d_CrCb_middle, d_Cr, sizeof(uchar) * WIDTH_HALF * HEIGHT_HALF / 2, cudaMemcpyDeviceToDevice);
        // checkError(err, "cudaMemcpy");

        // err = cudaMemcpy(h_Y, d_Y, sizeof(uchar) * WIDTH * HEIGHT, cudaMemcpyDeviceToHost);
        // checkError(err, "cudaMemcpy");
        err = cudaMemcpy(h_CrCb, d_CrCb, sizeof(uchar) * WIDTH_HALF * HEIGHT_HALF, cudaMemcpyDeviceToHost);
        checkError(err, "cudaMemcpy");
        err = cudaMemcpy(h_bgr, d_bgr, sizeof(uchar) * WIDTH * HEIGHT * CHANNEL, cudaMemcpyDeviceToHost);
        checkError(err, "cudaMemcpy");
        // err = cudaMemcpy(h_Cr, d_Cr, sizeof(uchar) * WIDTH_HALF * HEIGHT_HALF / 2, cudaMemcpyDeviceToHost);
        // checkError(err, "cudaMemcpy");
        // err = cudaMemcpy(h_Cb, d_Cb, sizeof(uchar) * WIDTH_HALF * HEIGHT_HALF / 2, cudaMemcpyDeviceToHost);
        // checkError(err, "cudaMemcpy");

        // dumpArrayFile(h_Y, WIDTH * HEIGHT, "Y", FILE_IDX);
        dumpArrayFile(h_CrCb, WIDTH_HALF * HEIGHT_HALF, "CrCb", FILE_IDX);
        dumpArrayFile(h_bgr, WIDTH * HEIGHT * CHANNEL, "bgr", FILE_IDX);
        // dumpArrayFile(h_Cr, WIDTH_HALF * HEIGHT_HALF / 2, "Cr", FILE_IDX);
        // dumpArrayFile(h_Cb, WIDTH_HALF * HEIGHT_HALF / 2, "Cb", FILE_IDX);
        FILE_IDX += 1;

#else
        /**
         *
         * CUDA Array -> Linear (Device to Host)
         *
         */

        // unsigned char *h_Y = (unsigned char *)malloc(ARRAY_BYTES);
        // unsigned char *h_CrCb = (unsigned char *)malloc(ARRAY_BYTES / 2);

        // err = cudaMemcpy2DFromArray(h_Y,
        //                             WIDTH * sizeof(unsigned char),
        //                             (cudaArray_t)cuY,
        //                             0,
        //                             0,
        //                             WIDTH * sizeof(unsigned char),
        //                             HEIGHT,
        //                             cudaMemcpyDeviceToHost);

        // checkError(err, "cudaMemcpy2DFromArray");

        // err = cudaMemcpy2DFromArray(h_CrCb,
        //                             WIDTH * sizeof(unsigned char),
        //                             (cudaArray_t)cuCrCb,
        //                             0,
        //                             0,
        //                             WIDTH * sizeof(unsigned char),
        //                             (HEIGHT / 2),
        //                             cudaMemcpyDeviceToHost);

        // checkError(err, "cudaMemcpy2DFromArray");

        // cv::Mat cpuYMat;
        // cpuYMat.create(HEIGHT, WIDTH, CV_8U);
        // cpuYMat.data = h_Y;
        // cpuYMat.step = WIDTH;

        // cv::Mat cpuCrCbMat;
        // cpuCrCbMat.create((HEIGHT / 2), WIDTH, CV_8U);
        // cpuCrCbMat.data = h_CrCb;
        // cpuCrCbMat.step = WIDTH;

        // // std::cout << cpuCrCbMat <<std::endl;
        // // std::cout << std::endl;

        // // cv::Mat cpuNV21;
        // // cv::cvtColorTwoPlane(cpuYMat, cpuCrCbMat, cpuNV21, COLOR_YUV2BGR_NV12);

        // cv::imshow("Y", cpuYMat);
        // cv::imshow("CrCb", cpuCrCbMat);
        // // cv::imshow("NV21", cpuNV21);
        // cv::waitKey(1);

        // 2) CUDA Array -> Linear (Device to Host)
        unsigned char *h_Y = (unsigned char *)malloc(ARRAY_BYTES);
        unsigned char *h_CrCb = (unsigned char *)malloc(ARRAY_BYTES / 4);

        err = cudaMemcpy2DFromArray(h_Y,
                                    WIDTH * sizeof(unsigned char),
                                    (cudaArray_t)cuY,
                                    0,
                                    0,
                                    WIDTH * sizeof(unsigned char),
                                    HEIGHT,
                                    cudaMemcpyDeviceToHost);

        checkError(err, "cudaMemcpy2DFromArray");

        err = cudaMemcpy2DFromArray(h_CrCb,
                                    (WIDTH / 2) * sizeof(unsigned char),
                                    (cudaArray_t)cuCrCb,
                                    0,
                                    0,
                                    (WIDTH / 2) * sizeof(unsigned char),
                                    (HEIGHT / 2),
                                    cudaMemcpyDeviceToHost);

        checkError(err, "cudaMemcpy2DFromArray");

        cv::Mat cpuYMat;
        cpuYMat.create(HEIGHT, WIDTH, CV_8U);
        cpuYMat.data = h_Y;
        cpuYMat.step = WIDTH;

        cv::Mat cpuCrCbMat;
        cpuCrCbMat.create((HEIGHT / 2), (WIDTH / 2), CV_8U);
        cpuCrCbMat.data = h_CrCb;
        cpuCrCbMat.step = (WIDTH / 2);

        // std::cout << cpuCrCbMat <<std::endl;
        // std::cout << std::endl;

        cv::Mat cpuNV12;
        cv::cvtColorTwoPlane(cpuYMat, cpuCrCbMat, cpuNV12, COLOR_YUV2BGR_NV12);

        /// NV21 -> BGR
        // cv::Mat cpuBGR;

        printf("%d\n", cpuNV12.channels());
        printf("%d\n", cpuNV12.rows);
        printf("%d\n", cpuNV12.cols);

        cv::imshow("Y", cpuYMat);
        cv::imshow("CrCb", cpuCrCbMat);
        cv::imshow("NV12", cpuNV12);
        cv::waitKey(1);

#endif

        // free(h_output);

        // CUresult r;

        // // Create surface from luminance channel.
        // CUDA_RESOURCE_DESC cudaResourceDesc;
        // memset(&cudaResourceDesc, 0, sizeof(cudaResourceDesc));
        // cudaResourceDesc.resType = CU_RESOURCE_TYPE_ARRAY;
        // cudaResourceDesc.res.array.hArray = m_frame.frame.pArray[0];
        // CUsurfObject cudaArrayYPlane = 0;

        // r = cuSurfObjectCreate(&cudaArrayYPlane, &cudaResourceDesc);

        // if (r != CUDA_SUCCESS)
        // {
        //     ORIGINATE_ERROR("Unable to create the surface object (CUresult %s)",
        //                     getCudaErrorString(r));
        // }

        // cudaResourceDesc.res.array.hArray = m_frame.frame.pArray[1];
        // CUsurfObject cudaSurfObj2 = 0;
        // cuResult = cuSurfObjectCreate(&cudaSurfObj2, &cudaResourceDesc);
        // if (cuResult != CUDA_SUCCESS)
        // {
        //     ORIGINATE_ERROR("Unable to create surface object 2 (%s)", getCudaErrorString(cuResult));
        // }

        // size_t numElem = cudaEgl->planeDesc[0].pitch * cudaEgl->planeDesc[0].height;

        // size_t width = m_frame.width;
        // size_t height = m_frame.height;
        // size_t pith = m_frame.pitch;
        // size_t planeCount = m_frame.planeCount;

        // printf("===\n");
        // printf("width: %u\n", width);
        // printf("height: %u\n", height);
        // printf("pith: %u\n", pith);
        // printf("planeCount: %u\n", planeCount);

        // size_t ARRAY_SIZE = width * height;
        // size_t ARRAY_BYTES = sizeof(uchar) * ARRAY_SIZE;

        // CUarray cuArray = m_frame.frame.pArray[0];
        // // uchar *d_ptr = (uchar *)m_frame.frame.pPitch[0];
        // uchar *h_ptr = (uchar *)malloc(ARRAY_BYTES);

        // printf("%p\n", &cuArray);
        // // printf("%p\n", d_ptr);
        // printf("%p\n", h_ptr);

        // cv::cuda::GpuMat()

        // printf("%u\n", ((uchar*)cuArray)[0]);

        // for (int i = 0; i < 10; i++)
        // {
        //     printf("%u ", ((uchar*)cuArray)[i]);
        // }
        // printf("\n");

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
        //         printf("%u ", ((uchar*) cuArray)[idx]);
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