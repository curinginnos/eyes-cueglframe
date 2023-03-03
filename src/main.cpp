/*
 * Copyright (c) 2016 - 2019, NVIDIA CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#include "main.hpp"
#include "StereoDisparityConsumerThread.hpp"

using namespace Argus;

namespace ArgusSamples
{
    // Debug print macros.
    #define PRODUCER_PRINT(...) printf("PRODUCER: " __VA_ARGS__)
    #define CONSUMER_PRINT(...) printf("CONSUMER: " __VA_ARGS__)


    /*
     * This sample opens a session with two sensors, it then using CUDA computes the histogram
     * of the sensors and computes a KL distance between the two histograms. A small value near
     * 0 indicates that the two images are alike. The processing of the images happens in the worker
     * thread of StereoDisparityConsumerThread. While the main app thread is used to drive the captures.
     */

    static bool execute(const CommonOptions &options)
    {
        // Initialize EGL.
        PROPAGATE_ERROR(g_display.initialize());

        // Initialize the Argus camera provider.
        UniqueObj<CameraProvider> cameraProvider(CameraProvider::create());

        // Get the ICameraProvider interface from the global CameraProvider.
        ICameraProvider *iCameraProvider = interface_cast<ICameraProvider>(cameraProvider);
        if (!iCameraProvider)
            ORIGINATE_ERROR("Failed to get ICameraProvider interface");
        printf("Argus Version: %s\n", iCameraProvider->getVersion().c_str());

        // Get the camera devices.
        std::vector<CameraDevice *> cameraDevices;
        iCameraProvider->getCameraDevices(&cameraDevices);
        // if (cameraDevices.size() < 2)
        //     ORIGINATE_ERROR("Must have at least 2 sensors available");

        std::vector<CameraDevice *> lrCameras;
        lrCameras.push_back(cameraDevices[0]); // Left Camera (the 1st camera will be used for AC)
        // lrCameras.push_back(cameraDevices[0]); // Right Camera

        // Create the capture session, AutoControl will be based on what the 1st device sees.
        UniqueObj<CaptureSession> captureSession(iCameraProvider->createCaptureSession(lrCameras));
        ICaptureSession *iCaptureSession = interface_cast<ICaptureSession>(captureSession);
        if (!iCaptureSession)
            ORIGINATE_ERROR("Failed to get capture session interface");

        // Create stream settings object and set settings common to both streams.
        UniqueObj<OutputStreamSettings> streamSettings(
            iCaptureSession->createOutputStreamSettings(STREAM_TYPE_EGL));
        IOutputStreamSettings *iStreamSettings = interface_cast<IOutputStreamSettings>(streamSettings);
        IEGLOutputStreamSettings *iEGLStreamSettings =
            interface_cast<IEGLOutputStreamSettings>(streamSettings);
        if (!iStreamSettings || !iEGLStreamSettings)
            ORIGINATE_ERROR("Failed to create OutputStreamSettings");
        iEGLStreamSettings->setMetadataEnable(true);
        iEGLStreamSettings->setPixelFormat(PIXEL_FMT_YCbCr_420_888);
        iEGLStreamSettings->setResolution(STREAM_SIZE);
        iEGLStreamSettings->setEGLDisplay(g_display.get());

        // Create egl streams
        PRODUCER_PRINT("Creating left stream.\n");
        iStreamSettings->setCameraDevice(lrCameras[0]);
        UniqueObj<OutputStream> streamLeft(iCaptureSession->createOutputStream(streamSettings.get()));
        IEGLOutputStream *iStreamLeft = interface_cast<IEGLOutputStream>(streamLeft);
        if (!iStreamLeft)
            ORIGINATE_ERROR("Failed to create left stream");

        PRODUCER_PRINT("Creating right stream.\n");
        iStreamSettings->setCameraDevice(lrCameras[0]);
        UniqueObj<OutputStream> streamRight(iCaptureSession->createOutputStream(streamSettings.get()));
        IEGLOutputStream *iStreamRight = interface_cast<IEGLOutputStream>(streamRight);
        if (!iStreamRight)
            ORIGINATE_ERROR("Failed to create right stream");

        PRODUCER_PRINT("Launching disparity checking consumer\n");
        StereoDisparityConsumerThread disparityConsumer(iStreamLeft, iStreamRight);
        PROPAGATE_ERROR(disparityConsumer.initialize());
        PROPAGATE_ERROR(disparityConsumer.waitRunning());

        // Create a request
        UniqueObj<Request> request(iCaptureSession->createRequest());
        IRequest *iRequest = interface_cast<IRequest>(request);
        if (!iRequest)
            ORIGINATE_ERROR("Failed to create Request");

        // Enable both streams in the request.
        iRequest->enableOutputStream(streamLeft.get());
        iRequest->enableOutputStream(streamRight.get());

        // Submit capture for the specified time.
        PRODUCER_PRINT("Starting repeat capture requests.\n");
        if (iCaptureSession->repeat(request.get()) != STATUS_OK)
            ORIGINATE_ERROR("Failed to start repeat capture request for preview");
        sleep(options.captureTime());

        // Stop the capture requests and wait until they are complete.
        iCaptureSession->stopRepeat();
        iCaptureSession->waitForIdle();

        // Disconnect Argus producer from the EGLStreams (and unblock consumer acquire).
        PRODUCER_PRINT("Captures complete, disconnecting producer.\n");
        iStreamLeft->disconnect();
        iStreamRight->disconnect();

        // Wait for the consumer thread to complete.
        PROPAGATE_ERROR(disparityConsumer.shutdown());

        // Shut down Argus.
        cameraProvider.reset();

        // Cleanup the EGL display
        PROPAGATE_ERROR(g_display.cleanup());

        PRODUCER_PRINT("Done -- exiting.\n");
        return true;
    }

}; // namespace ArgusSamples

int main(int argc, char *argv[])
{
    ArgusSamples::CommonOptions options(basename(argv[0]),
                                        ArgusSamples::CommonOptions::Option_T_CaptureTime);
    if (!options.parse(argc, argv))
        return EXIT_FAILURE;
    if (options.requestedExit())
        return EXIT_SUCCESS;

    if (!ArgusSamples::execute(options))
        return EXIT_FAILURE;

    return EXIT_SUCCESS;
}
