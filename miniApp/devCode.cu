// Copyright 2025 Ingo Wald
// SPDX-License-Identifier: Apache-2.0

#include "rafi/rafi.h"
#include "miniApp.h"

DECLARE_OPTIX_LAUNCH_PARAMS(miniApp::PerLaunchData);

namespace miniApp {

  /*! this works just like a cuda kernel, you just can't directly pass
      any paramters; they have to go through the "LaunchParams"
      abstraction */
  OPTIX_RAYGEN_PROGRAM(miniApp_perPixel)()
  {
    /* TODO */
  }

}
