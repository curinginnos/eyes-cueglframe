#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "CommonOptions.h"
#include "CUDAHelper.h"
#include "EGLGlobal.h"
#include "Error.h"

#include <iostream>


#include <Argus/Argus.h>


namespace ArgusSamples
{
    // Constants.
    static const Argus::Size2D<uint32_t> STREAM_SIZE(640, 480);

    // Globals and derived constants.
    EGLDisplayHolder g_display;

}