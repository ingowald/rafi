#pragma once

namespace rafi {

  template<typename ray_t>
  struct DevContext {
    /*! number of rays that were sent to this rank */
    int numIncomingRays();

    const ray_t *getIncomingRay(int rayID) const;
    
    /*! put specified ray into the out-queue, and mark it for going to
        given rank */
    void emitOutgoingRay(const ray_t ray, int rankThisNeedsToGetSentTo);
  };
  
  template<typename RayT>
  struct HostContext
  {
    struct ForwardResult {
      int numRaysAliveAcrossAllRanks;
      int numRaysInIncomingQueueThisRank;
    };
    
    HostContext(MPI_Comm comm);
    virtual ~Context() = default;

    /*! allocates given number of rays in internal buffers on each
        node. app guarantees that no ray will ever generate or receive
        more rays than indicates in this function */
    virtual void resizeRayQueues(size_t maxRaysOnAnyRankAtAnyTime) = 0;

    /*! forward current set of (one-per-rank) outgoing ray queues,
        such that each ray ends up in the incoming ray queue on the
        rank it specified during its `emitOutoingRay()` call. This
        call is collaborative blocking; all ranks in the communicator
        have to call it (even if that rank has no outgoing rays), and
        ranks will block until all rays have been delivered. Return
        value indicates both how many rays this rank just had incoming
        in this forwarding operation AND the total number of rays
        currently in flight ACROSS ALL RANKS (which can be used for
        distributed termination). */
    virtual ForwardResult forwardRays() = 0;
  };
  
  /*! creates a new rafi context over the given mpi communicator. all
    ranks in the given comm need to call this method, the call will
    be blocking. */
  template<typename RayT>
  HostContext<RayT> *createContext(MPI_Comm comm);
  
};
