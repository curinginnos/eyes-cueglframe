#include "opencv2/core.hpp"

// __global__ void _unravel(uchar* src, uchar* dst1, uchar* dst2, int pitch, int n);
void unravel(uchar* src, uchar* dst1, uchar* dst2, int WIDTH, int HEIGHT);