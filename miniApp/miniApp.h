#pragma once

#include "rafi/rafi.h"

namespace miniApp {
  
  struct Ray {
    int srcRank;
    int srcID;
  };
  
  struct PerRayData {
  };
  
  struct PerLaunchData {
    rafi::DeviceInterface<Ray> rafi;
  };
  
}

