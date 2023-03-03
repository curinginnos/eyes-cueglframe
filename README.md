# Argus + CUDA

## Overview

This project is based on `argus/samples/syncSensor`.

`CUstream`, a CUDA version of `EGLStream`, is introduced in the original code, but doesn't provide the conversion from `CUeglFrame` into `cv::cuda::GpuMat`

## Prerequisite

### YUV

YUV is a a color model, just like BGR. What's special about YUV is its compression method.

You might have seen YUV444, YUV422, YUV420, and YCrC420.

BTW, YCrCb and YUV are the same.

Three digits that come after denotes how it's compressed.

TBA ... 