#pragma once

#include <owl/owl.h>
#include <mpi.h>

namespace rafi {

  template<typename ray_t> struct RafiImpl;
  
  template<typename ray_t>
  struct DeviceInterface {
    /*! number of rays that were sent to this rank */
    inline __device__
    int numIncoming() const;

    inline __device__
    ray_t getIncoming(int rayID) const;

    /*! append the provided ray to the out-queue, and mark it for
      being sent to specified destination rank. Destination must be
      a valid mpi rank for the communicator being provided (though
      sending to itself is allowed */
    inline __device__
    void emitOutgoing(ray_t ray, int destination) const;

    inline __device__
    int allocateOutgoing(int numRaysToAllocate) const;
    inline __device__
    void writeOutgoing(int rayID, ray_t ray, int destination) const;

    struct {
      int rank;
      int size;
    } mpi;
  private:
    friend class RafiImpl<ray_t>;
    int m_numIncoming;
    
    /*! pointer to atomic int for counting outgoing rays */
    int *pNumOutgoing;
    
    /*! max number of rays reserved for outgoing ray. asking for more
      is a error and will return null */
    int maxNumOutgoing;

    int2  *pDestOut;
    ray_t *pRaysOut;
    ray_t *pRaysIn;
  };
  
  struct ForwardResult {
    int numRaysAliveAcrossAllRanks;
    int numRaysInIncomingQueueThisRank;
  };
    
  template<typename RayT>
  struct HostContext
  {
    virtual ~HostContext() = default;

    virtual DeviceInterface<RayT> getDeviceInterface() = 0;
    
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

    struct {
      int rank = -1;
      int size = -1;
      MPI_Comm comm = MPI_COMM_NULL;
    } mpi;
  };
  
  /*! creates a new rafi context over the given mpi communicator. all
    ranks in the given comm need to call this method, the call will
    be blocking. */
  template<typename RayT>
  HostContext<RayT> *createContext(MPI_Comm comm);




  // ==================================================================
  // device inline implementations
  // ==================================================================

#ifdef __CUDACC__
  /*! number of rays that were sent to this rank */
  template<typename ray_t>
  inline __device__
  int DeviceInterface<ray_t>::numIncoming() const
  { return m_numIncoming; }
  
  template<typename ray_t>
  inline __device__
  ray_t DeviceInterface<ray_t>::getIncoming(int rayID) const
  {
    return pRaysIn[rayID];
  }
  
  template<typename ray_t>
  inline __device__
  int DeviceInterface<ray_t>::allocateOutgoing(int numRaysOutgoing) const
  {
    return atomicAdd(pNumOutgoing,numRaysOutgoing);
  }
  
  template<typename ray_t>
  inline __device__
  void DeviceInterface<ray_t>::emitOutgoing(const ray_t ray,
                                            int rankThisNeedsToGetSentTo) const
  {
    writeOutgoing(allocateOutgoing(1),ray,rankThisNeedsToGetSentTo);
  }

  template<typename ray_t>
  inline __device__
  void DeviceInterface<ray_t>::writeOutgoing(int rayID, ray_t ray, int destination) const
  {
    pRaysOut[rayID] = ray;
    pDestOut[rayID] = make_int2(rayID,destination);
  }
  
#endif
  
};
