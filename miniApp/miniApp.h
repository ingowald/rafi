#pragma once

#include "rafi/rafi.h"

namespace miniApp {
  
struct Ray {
  int foo;
};

struct PerRayData {
};

struct PerLaunchData {
  rafi::DeviceInterface<Ray> rafi;
};

}

