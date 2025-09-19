#include "miniApp.h"
#include "rafi/implementation.h"

RAFI_INSTANTIATE(miniApp::Ray)

namespace miniApp {
  using namespace rafi;

  __global__
  void generateRays(rafi::DeviceInterface<Ray> rafi,
                    int numRays)
  {
    int tid = threadIdx.x+blockIdx.x*blockDim.x;
    if (tid >= numRays) return;
    
    Ray ray;
    ray.srcRank = rafi.mpi.rank;
    ray.srcID = tid;

    int dst = (123+13*17*23*tid) % (2*rafi.mpi.size);
    if (dst < rafi.mpi.size)
      rafi.emitOutgoing(ray,dst);
  }

  __global__
  void processRays(int round,
                   rafi::DeviceInterface<Ray> rafi,
                   int numRays)
  {
    int tid = threadIdx.x+blockIdx.x*blockDim.x;
    if (tid >= numRays) return;
    
    Ray ray = rafi.getIncoming(tid);

    int dst = (1234+(round+1)*(13+17*(ray.srcID+23*tid))) % (2*rafi.mpi.size);
    if (tid > 1 && dst < rafi.mpi.size)
      rafi.emitOutgoing(ray,dst);
  }

  
  void run(int ac, char **av)
  {
    RAFI_CUDA_CALL(Free(0));
    RAFI_MPI_CALL(Init(&ac,&av));
    
    rafi::HostContext<Ray> *rafi = rafi::createContext<Ray>(MPI_COMM_WORLD);

    int maxRays = 1000*128*rafi->mpi.size;
    rafi->resizeRayQueues(maxRays);

    srand48(rafi->mpi.rank);
    int numRaysSeed = int(10+128*drand48())*(128);

    {
      int bs = 1024;
      int nb = divRoundUp(numRaysSeed,bs);
      generateRays<<<nb,bs>>>(rafi->getDeviceInterface(),numRaysSeed);
      RAFI_CUDA_SYNC_CHECK();
    }

    for (int round=0;true;round++) {
      PRINT(round);
      rafi::ForwardResult result = rafi->forwardRays();
      if (result.numRaysAliveAcrossAllRanks == 0)
        break;
      int numRays = result.numRaysInIncomingQueueThisRank;
      PRINT(result.numRaysInIncomingQueueThisRank);
      int bs = 1024;
      int nb = divRoundUp(numRays,bs);
      processRays<<<nb,bs>>>(round,rafi->getDeviceInterface(),numRays);
    }
    
    delete rafi;
    
    RAFI_MPI_CALL(Finalize());
  }
}

int main(int ac, char **av)
{
  miniApp::run(ac,av);
  return 0;
}

