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

    void dumpImageFile(uchar *arr, const char *name, int idx, int width, int height)
    {
        char filename[50];

        sprintf(filename, "%s_%d.png", name, idx);

        cv::Mat frame;
        frame.create(height, width, CV_8U);
        frame.data = arr;

        cv::imwrite(filename, frame);
    }

    int FILE_IDX = 0;

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


    size_t calcByte(){

    }

    bool ScopedCudaEGLStreamFrameAcquire::cvtYUV2BGR() const
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


        uchar *d_Y;
        uchar *d_CrCb;
        cudaMalloc(&d_Y, sizeof(uchar) * WIDTH * HEIGHT);
        cudaMalloc(&d_CrCb, sizeof(uchar) * WIDTH_HALF * HEIGHT_HALF);

        // Retrieve Y and CbCr palnes
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


        // Unravel CbCr into Cb and Cr
        uchar *d_Cb;
        uchar *d_Cr;
        cudaMalloc(&d_Cb, sizeof(uchar) * WIDTH_HALF * HEIGHT_HALF);
        cudaMalloc(&d_Cr, sizeof(uchar) * WIDTH_HALF * HEIGHT_HALF);
        unravel(d_CrCb, d_Cb, d_Cr, WIDTH_HALF, HEIGHT_HALF);

        uchar *h_Y = (uchar *) malloc(sizeof(uchar) * WIDTH * HEIGHT);
        uchar *h_Cb = (uchar *) malloc(sizeof(uchar) * WIDTH * HEIGHT);
        uchar *h_Cr = (uchar *) malloc(sizeof(uchar) * WIDTH * HEIGHT);

        cudaMemcpy(h_Y, d_Y, sizeof(uchar) * WIDTH * HEIGHT, cudaMemcpyDeviceToHost);
        cudaMemcpy(h_Cb, d_Cb, sizeof(uchar) * WIDTH_HALF_HALF * HEIGHT_HALF_HALF, cudaMemcpyDeviceToHost);
        cudaMemcpy(h_Cr, d_Cr, sizeof(uchar) * WIDTH_HALF_HALF * HEIGHT_HALF_HALF, cudaMemcpyDeviceToHost);

        cv::Mat matY;
        matY.create(HEIGHT, WIDTH, CV_8U);
        matY.data = h_Y;

        cv::Mat matCb;
        matCb.create(HEIGHT_HALF_HALF, WIDTH_HALF_HALF, CV_8U);
        matCb.data = h_Cb;

        cv::Mat matCr; 
        matCr.create(HEIGHT_HALF_HALF, WIDTH_HALF_HALF, CV_8U);
        matCr.data = h_Cr;

        cv::imshow("matY", matY);
        cv::imshow("matCb", matCb);
        cv::imshow("matCr", matCr);
        cv::pollKey();

        cudaFree(d_Y);
        cudaFree(d_CrCb);
        cudaFree(d_Cr);
        cudaFree(d_Cb);
        free(h_Y);
        free(h_Cb);
        free(h_Cr);

        return true;
    }

    bool ScopedCudaEGLStreamFrameAcquire::cvtCUDAYUV2BGR() const
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

        uchar *d_Y;
        uchar *d_CrCb;
        cudaMalloc(&d_Y, sizeof(uchar) * WIDTH * HEIGHT);
        cudaMalloc(&d_CrCb, sizeof(uchar) * WIDTH_HALF * HEIGHT_HALF);

        // Retrieve Y and CbCr palnes
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

        uchar *d_YCrCb;
        cudaMalloc(&d_YCrCb, sizeof(uchar) * WIDTH * (HEIGHT + HEIGHT_HALF_HALF));
    
        uchar *ptr = d_YCrCb;

        err = cudaMemcpy(ptr, d_Y, sizeof(uchar) * WIDTH * HEIGHT, cudaMemcpyDeviceToDevice);
        checkError(err, "cudaMemcpy - Copying d_Y to d_YCrCb");

        ptr += (WIDTH * HEIGHT);

        err = cudaMemcpy(ptr, d_CrCb, sizeof(uchar) * WIDTH_HALF * HEIGHT_HALF, cudaMemcpyDeviceToDevice);
        checkError(err, "cudaMemcpy - Copying d_CrCb to d_YCrCb");


        cv::cuda::GpuMat gpuMatYUV420sp;
        gpuMatYUV420sp.create((HEIGHT + HEIGHT_HALF_HALF), WIDTH, CV_8U);
        gpuMatYUV420sp.data = d_YCrCb;
        gpuMatYUV420sp.step = WIDTH * sizeof(uchar);

        cv::cuda::GpuMat gpuMatBGR;
        gpuMatBGR.create(HEIGHT, WIDTH, CV_8UC3);
        gpuMatBGR.step = WIDTH * 3 * sizeof(uchar);

        // CPU works, but CUDA doesn't support 
        cv::cuda::cvtColor(gpuMatYUV420sp, gpuMatBGR, cv::COLOR_YUV420sp2BGR);
        
        cv::Mat cpuMatBGR;
        cpuMatBGR.create(HEIGHT, WIDTH, CV_8UC3);
        cpuMatBGR.step = WIDTH * 3 * sizeof(uchar);

        gpuMatBGR.download(cpuMatBGR);

        cudaFree(d_Y);
        cudaFree(d_CrCb);

        return true;
    }

    bool ScopedCudaEGLStreamFrameAcquire::cvtNPPYUV2BGR() const
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

        // uchar *d_CrCb_linear;
        // cudaMalloc(&d_CrCb_linear, sizeof(uchar) * WIDTH_HALF * HEIGHT_HALF);

        // err = cudaMemcpy(d_CrCb_linear, d_CrCb, WIDTH_HALF * HEIGHT_HALF * sizeof(uchar), cudaMemcpyDeviceToDevice);

        // checkError(err, "cudaMemcpy2DFromArray - CrCb_linear");


        // printf("%x\n", m_frame.eglColorFormat);
        // printf("%d\n", m_frame.planeCount);


        // invalid argument
        // err = cudaMemcpy2DFromArray(d_CrCb,
        //                             WIDTH * sizeof(uchar),
        //                             (cudaArray_t)cuCrCb,
        //                             0,
        //                             0,
        //                             WIDTH_HALF * sizeof(uchar),
        //                             HEIGHT_HALF,
        //                             cudaMemcpyDeviceToDevice);

        // invalid argument
        // err = cudaMemcpy(d_CrCb, ptrCrCb, WIDTH_HALF * HEIGHT_HALF * sizeof(uchar), cudaMemcpyDeviceToDevice);

        // checkError(err, "cudaMemcpy2DFromArray - CrCb");


        // {
        //     Npp8u *const pSrc[2] = {d_Y, d_CrCb};
        //     int rSrcStep = WIDTH * sizeof(uchar);
        //     int nDstStep = WIDTH * 4 * sizeof(uchar);
        //     NppiSize oSizeROI = {WIDTH, HEIGHT};

        //     uchar *d_bgr;
        //     cudaMalloc(&d_bgr, sizeof(uchar) * WIDTH * HEIGHT * 4);

        //     nppErr = nppiNV21ToBGR_8u_P2C4R(pSrc, rSrcStep, d_bgr, nDstStep, oSizeROI);

        //     if(nppErr)
        //     {
        //         printf("%d\n", nppErr);
        //     }

        //     cv::cuda::GpuMat gpuMatBGRA;
        //     gpuMatBGRA.create(HEIGHT, WIDTH, CV_8UC4);
        //     gpuMatBGRA.data = d_bgr;
        //     gpuMatBGRA.step = WIDTH * 4 * sizeof(uchar);

        //     cv::Mat cpuMatBGRA;
        //     cpuMatBGRA.create(HEIGHT, WIDTH, CV_8UC4);
        //     cpuMatBGRA.step = WIDTH * 4 * sizeof(uchar);

        //     gpuMatBGRA.download(cpuMatBGRA);


        //     cv::imshow("img", cpuMatBGRA);
        //     cv::pollKey();
        // }



        {
            // cv::cuda::GpuMat gpuMatCrCb_linear;
            // gpuMatCrCb_linear.create(HEIGHT_HALF, WIDTH_HALF, CV_8U);
            // gpuMatCrCb_linear.data = d_CrCb_linear;
            // gpuMatCrCb_linear.step = WIDTH_HALF  * sizeof(uchar);

            // cv::Mat cpuMatCrCb_linear;
            // cpuMatCrCb_linear.create(HEIGHT_HALF, WIDTH_HALF, CV_8U);
            // cpuMatCrCb_linear.step = WIDTH_HALF  * sizeof(uchar);

            // gpuMatCrCb_linear.download(cpuMatCrCb_linear);

            // cv::cuda::GpuMat gpuMatCrCb;
            // gpuMatCrCb.create(HEIGHT_HALF, WIDTH_HALF, CV_8U);
            // gpuMatCrCb.data = d_CrCb;
            // gpuMatCrCb.step = WIDTH_HALF  * sizeof(uchar);

            // cv::Mat cpuMatCrCb;
            // cpuMatCrCb.create(HEIGHT_HALF, WIDTH_HALF, CV_8U);
            // cpuMatCrCb.step = WIDTH_HALF  * sizeof(uchar);

            // gpuMatCrCb.download(cpuMatCrCb);


            // cv::imshow("cpuMatCrCb_linear", cpuMatCrCb_linear);
            // cv::imshow("cpuMatCrCb", cpuMatCrCb);
            // cv::pollKey();            

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

            // cv::Mat cpuMatBGR;
            // cv::cvtColor(cpuMatBGRA, cpuMatBGR, cv::COLOR_BGRA2BGR);

            cv::imshow("img", cpuMatBGR);
            cv::pollKey();
        }   

        // {
        //     cv::cuda::GpuMat gpuMatCrCb;
        //     gpuMatCrCb.create(HEIGHT_HALF, WIDTH, CV_8U);
        //     gpuMatCrCb.data = d_CrCb;
        //     gpuMatCrCb.step = WIDTH  * sizeof(uchar);

        //     cv::Mat cpuMatCrCb;
        //     cpuMatCrCb.create(HEIGHT_HALF, WIDTH, CV_8UC4);
        //     cpuMatCrCb.step = WIDTH  * sizeof(uchar);

        //     gpuMatCrCb.download(cpuMatCrCb);


        //     cv::imshow("img", cpuMatCrCb);
        //     cv::pollKey();            
        // }     

        // {
        //     // Resize Cb and Cr to match OpenCV's YUV420 format
        //     double nXFactor = 2.0;
        //     double nYFactor = 1.0;
        //     uchar *src_resize;
        //     int resize_len = (int)round(WIDTH * HEIGHT_HALF / 2);


        //     uchar *d_CrCb_resized;
        //     cudaMalloc(&d_CrCb_resized, sizeof(uchar) * WIDTH * HEIGHT_HALF);

        //     NppiSize oSrcSize = {WIDTH_HALF, HEIGHT_HALF};
        //     NppiSize oDstSize = {WIDTH_HALF * nXFactor, HEIGHT_HALF * nYFactor};
        //     NppiRect oSrcROI = {0, 0, WIDTH_HALF, HEIGHT_HALF};
        //     NppiRect oDstROI = {0, 0, (WIDTH_HALF * nXFactor), (HEIGHT_HALF * nYFactor)};

        //     NppStatus nppStatus;

        //     nppStatus = nppiResize_8u_C1R(d_CrCb,
        //                                 oSrcSize.width * sizeof(uchar),
        //                                 oSrcSize,
        //                                 oSrcROI,
        //                                 d_CrCb_resized,
        //                                 oDstSize.width * sizeof(uchar),
        //                                 oDstSize,
        //                                 oDstROI,
        //                                 NPPI_INTER_NN);

        //     checkNppStatus(nppStatus, "Resizing Cr plane");


        //     Npp8u *const pSrc[2] = {d_Y, d_CrCb_resized};
        //     int rSrcStep = WIDTH * sizeof(uchar);
        //     int nDstStep = WIDTH * 3 * sizeof(uchar);
        //     NppiSize oSizeROI = {WIDTH, HEIGHT};

        //     uchar *d_bgr;
        //     cudaMalloc(&d_bgr, sizeof(uchar) * WIDTH * HEIGHT * 3);

        //     nppErr = nppiNV12ToBGR_8u_P2C3R(pSrc, rSrcStep, d_bgr, nDstStep, oSizeROI);

        //     if(nppErr)
        //     {
        //         printf("%d\n", nppErr);
        //     }

        //     cv::cuda::GpuMat gpuMatBGRA;
        //     gpuMatBGRA.create(HEIGHT, WIDTH, CV_8UC3);
        //     gpuMatBGRA.data = d_bgr;
        //     gpuMatBGRA.step = WIDTH * 3 * sizeof(uchar);

        //     cv::Mat cpuMatBGR;
        //     cpuMatBGR.create(HEIGHT, WIDTH, CV_8UC3);
        //     cpuMatBGR.step = WIDTH * 3 * sizeof(uchar);

        //     gpuMatBGRA.download(cpuMatBGR);


        //     cv::imshow("img", cpuMatBGR);
        //     cv::pollKey();
        // }     


        // {
        //     // Resize Cb and Cr to match OpenCV's YUV420 format
        //     double nXFactor = 2.0;
        //     double nYFactor = 1.0;
        //     uchar *src_resize;
        //     int resize_len = (int)round(WIDTH * HEIGHT_HALF / 2);


        //     uchar *d_CrCb_resized;
        //     cudaMalloc(&d_CrCb_resized, sizeof(uchar) * WIDTH * HEIGHT_HALF);

        //     NppiSize oSrcSize = {WIDTH_HALF, HEIGHT_HALF};
        //     NppiSize oDstSize = {WIDTH_HALF * nXFactor, HEIGHT_HALF * nYFactor};
        //     NppiRect oSrcROI = {0, 0, WIDTH_HALF, HEIGHT_HALF};
        //     NppiRect oDstROI = {0, 0, (WIDTH_HALF * nXFactor), (HEIGHT_HALF * nYFactor)};

        //     NppStatus nppStatus;

        //     nppStatus = nppiResize_8u_C1R(d_CrCb,
        //                                 oSrcSize.width * sizeof(uchar),
        //                                 oSrcSize,
        //                                 oSrcROI,
        //                                 d_CrCb_resized,
        //                                 oDstSize.width * sizeof(uchar),
        //                                 oDstSize,
        //                                 oDstROI,
        //                                 NPPI_INTER_NN);

        //     checkNppStatus(nppStatus, "Resizing Cr plane");


        //     Npp8u *const pSrc[2] = {d_Y, d_CrCb_resized};
        //     int rSrcStep = WIDTH * sizeof(uchar);
        //     int nDstStep = WIDTH * 4 * sizeof(uchar);
        //     NppiSize oSizeROI = {WIDTH, HEIGHT};

        //     uchar *d_bgr;
        //     cudaMalloc(&d_bgr, sizeof(uchar) * WIDTH * HEIGHT * 4);

        //     nppErr = nppiNV21ToBGR_8u_P2C4R(pSrc, rSrcStep, d_bgr, nDstStep, oSizeROI);

        //     if(nppErr)
        //     {
        //         printf("%d\n", nppErr);
        //     }

        //     cv::cuda::GpuMat gpuMatBGRA;
        //     gpuMatBGRA.create(HEIGHT, WIDTH, CV_8UC4);
        //     gpuMatBGRA.data = d_bgr;
        //     gpuMatBGRA.step = WIDTH * 4 * sizeof(uchar);

        //     cv::Mat cpuMatBGRA;
        //     cpuMatBGRA.create(HEIGHT, WIDTH, CV_8UC4);
        //     cpuMatBGRA.step = WIDTH * 4 * sizeof(uchar);

        //     gpuMatBGRA.download(cpuMatBGRA);

        //     cv::Mat cpuMatBGR;
        //     cv::cvtColor(cpuMatBGRA, cpuMatBGR, cv::COLOR_BGRA2BGR);


        //     cv::imshow("img", cpuMatBGR);
        //     cv::pollKey();
        // }     

        // {
        //     // Resize Cb and Cr to match OpenCV's YUV420 format
        //     double nXFactor = 2.0;
        //     double nYFactor = 2.0;
        //     uchar *src_resize;
        //     // int resize_len = (int)round(WIDTH * HEIGHT / 2);


        //     uchar *d_CrCb_resized;
        //     cudaMalloc(&d_CrCb_resized, sizeof(uchar) * WIDTH * HEIGHT);

        //     NppiSize oSrcSize = {WIDTH_HALF, HEIGHT_HALF};
        //     NppiSize oDstSize = {WIDTH_HALF * nXFactor, HEIGHT_HALF * nYFactor};
        //     NppiRect oSrcROI = {0, 0, WIDTH_HALF, HEIGHT_HALF};
        //     NppiRect oDstROI = {0, 0, (WIDTH_HALF * nXFactor), (HEIGHT_HALF * nYFactor)};

        //     NppStatus nppStatus;

        //     nppStatus = nppiResize_8u_C1R(d_CrCb,
        //                                 oSrcSize.width * sizeof(uchar),
        //                                 oSrcSize,
        //                                 oSrcROI,
        //                                 d_CrCb_resized,
        //                                 oDstSize.width * sizeof(uchar),
        //                                 oDstSize,
        //                                 oDstROI,
        //                                 NPPI_INTER_NN);

        //     checkNppStatus(nppStatus, "Resizing Cr plane");


        //     Npp8u *const pSrc[2] = {d_Y, d_CrCb_resized};
        //     int rSrcStep = WIDTH * sizeof(uchar);
        //     int nDstStep = WIDTH * 3 * sizeof(uchar);
        //     NppiSize oSizeROI = {WIDTH, HEIGHT};

        //     uchar *d_bgr;
        //     cudaMalloc(&d_bgr, sizeof(uchar) * WIDTH * HEIGHT * 3);

        //     nppErr = nppiNV12ToBGR_8u_P2C3R(pSrc, rSrcStep, d_bgr, nDstStep, oSizeROI);

        //     if(nppErr)
        //     {
        //         printf("%d\n", nppErr);
        //     }

        //     cv::cuda::GpuMat gpuMatBGR;
        //     gpuMatBGR.create(HEIGHT, WIDTH, CV_8UC3);
        //     gpuMatBGR.data = d_bgr;
        //     gpuMatBGR.step = WIDTH * 3 * sizeof(uchar);

        //     cv::Mat cpuMatBGR;
        //     cpuMatBGR.create(HEIGHT, WIDTH, CV_8UC3);
        //     cpuMatBGR.step = WIDTH * 3 * sizeof(uchar);

        //     gpuMatBGR.download(cpuMatBGR);

        //     cv::imshow("img", cpuMatBGR);
        //     cv::pollKey();
        // }             

        cudaFree(d_Y);
        cudaFree(d_CrCb);
        
        return true;
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

    bool ScopedCudaEGLStreamFrameAcquire::generateHistogram(unsigned int histogramData[HISTOGRAM_BINS],
                                                            float *time)
    {
        if (!hasValidFrame() || !histogramData || !time)
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