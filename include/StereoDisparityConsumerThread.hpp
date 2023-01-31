#pragma once

#include <Argus/Argus.h>
#include <cudaEGL.h>

#include "CommonOptions.h"
#include "CUDAHelper.h"
#include "EGLGlobal.h"
#include "Error.h"

#include "Thread.h"

#include "../cudaHistogram/histogram.h"

namespace ArgusSamples
{

    // Debug print macros.
    #define PRODUCER_PRINT(...) printf("PRODUCER: " __VA_ARGS__)
    #define CONSUMER_PRINT(...) printf("CONSUMER: " __VA_ARGS__)


    /*******************************************************************************
     * Argus disparity class
     *   This class will analyze frames from 2 synchronized sensors and compute the
     *   KL distance between the two images. Large values of KL indicate a large disparity
     *   while a value of 0.0 indicates that the images are alike.
     ******************************************************************************/
    class StereoDisparityConsumerThread : public Thread
    {
    public:
        explicit StereoDisparityConsumerThread(Argus::IEGLOutputStream *leftStream,
                                               Argus::IEGLOutputStream *rightStream)
            : m_leftStream(leftStream), m_rightStream(rightStream), m_cudaContext(0), m_cuStreamLeft(NULL), m_cuStreamRight(NULL)
        {
        }
        ~StereoDisparityConsumerThread()
        {
        }

    private:
        /** @name Thread methods */
        /**@{*/
        virtual bool threadInitialize();
        virtual bool threadExecute();
        virtual bool threadShutdown();
        /**@}*/

        Argus::IEGLOutputStream *m_leftStream;
        Argus::IEGLOutputStream *m_rightStream;
        CUcontext m_cudaContext;
        CUeglStreamConnection m_cuStreamLeft;
        CUeglStreamConnection m_cuStreamRight;
    };
}