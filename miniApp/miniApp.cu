#include "miniApp.h"
#include "rafi/implementation.h"

RAFI_INSTANTIATE(miniApp::Ray)

namespace miniApp {
  void run(int ac, char **av)
  {
    RAFI_CUDA_CALL(Free(0));
    
    rafi::HostContext<Ray> *rafi = rafi::createContext<Ray>(MPI_COMM_WORLD);
    
    delete rafi;
  }
}

int main(int ac, char **av)
{
  miniApp::run(ac,av);
  return 0;
}

