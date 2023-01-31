# Argus + OpenCV + GPU

```sh
$ sudo systemctl start nvargus-daemon.service
```

```
nano1@nano1-desktop:/usr/local/cuda/bin$ sudo ./nvprof /usr/src/jetson_multimedia_api/argus/build/samples/syncSensor/argus_syncsensor 
```

`Argus` -> `EGL` -> `CUDA` -> `cv::GpuMat`

```
Loop begins
eglQueryStreamKHR
eglQueryStreamKHR
eglQueryStreamKHR finished
Frame acquired succesfully!
cuGraphicsResourceGetMappedEglFrame failed
unspecified launch failure
ScopedCudaEGL finished
0x7f844a0e50
HERE!
0x7f844a0e50
Before nppi
640
480
0x7f98d70aa0
0
After nppi
Failed
NPP_CUDA_KERNEL_EXECUTION_ERROR
```

## Reference
[Jetson MultiMedia API to cv::cuda::GpuMat](https://forums.developer.nvidia.com/t/jetson-multimedia-api-to-cv-gpumat/217292)
[​How can I transform CUeglFrame to OpenCV Mat object?](https://forums.developer.nvidia.com/t/how-can-i-transform-cueglframe-to-opencv-mat-object/164505)

##
```
width: 640
height: 480
pith: 0
planeCount: 2
```