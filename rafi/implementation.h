#pragma once

#include "rafi/rafi.h"
#include "rafi/cuda_check.h"
#include "rafi/mpi_check.h"
#include <cub/cub.cuh>

namespace rafi {

  /*! helper class to set active GPU for the lifetime of this class,
      and restore it to what it was before upon destruction of this
      class */
  struct SetActiveGPU {
    inline SetActiveGPU(int gpuID)
    {
      RAFI_CUDA_CHECK(cudaGetDevice(&savedActiveDeviceID));
      RAFI_CUDA_CHECK(cudaSetDevice(gpuID));
    }
    inline ~SetActiveGPU()
    { RAFI_CUDA_CALL_NOTHROW(SetDevice(savedActiveDeviceID)); }
  private:
    int savedActiveDeviceID = -1;
  };
  
  inline int divRoundUp(int a, int b) { return (a+b-1)/b; }

  template<typename ray_t>
  struct RafiImpl : public HostContext<ray_t>
  {
    RafiImpl(MPI_Comm comm, int gpuID);
    ~RafiImpl() override;
    void resizeRayQueues(size_t maxRaysOnAnyRankAtAnyTime) override;
    void clearQueue() override {
      SetActiveGPU forDuration(gpuID);
      cudaMemset(pNumOutgoing,0,sizeof(int));
    }

    DeviceInterface<ray_t> getDeviceInterface() override;
    ForwardResult forwardRays() override;

    int       numIncoming  = 0;
    int      *pNumOutgoing = 0;
    ray_t    *pRaysIn      = 0;
    ray_t    *pRaysOut     = 0;
    unsigned *pDestRank    = 0;
    unsigned *pDestRayID   = 0;
    int       numReserved  = 0;
    int       const gpuID;
    using HostContext<ray_t>::mpi;
  };
    
  template<typename ray_t>
  HostContext<ray_t> *createContext(MPI_Comm comm, int gpuID)
  {
    return new RafiImpl<ray_t>(comm, gpuID);
  }

  
  template<typename ray_t>
  RafiImpl<ray_t>::RafiImpl(MPI_Comm comm, int gpuID)
    : gpuID(gpuID)
  {
    mpi.comm = comm;
    MPI_Comm_rank(comm,&mpi.rank);
    MPI_Comm_size(comm,&mpi.size);

    RAFI_CUDA_CALL(Malloc((void**)&pNumOutgoing,sizeof(int)));
    RAFI_CUDA_CALL(Memset(pNumOutgoing,0,sizeof(int)));
  }
  
  template<typename ray_t>
  RafiImpl<ray_t>::~RafiImpl()
  {
    RAFI_CUDA_CALL_NOTHROW(Free(pNumOutgoing));
    RAFI_CUDA_CALL_NOTHROW(Free(pRaysIn));
    RAFI_CUDA_CALL_NOTHROW(Free(pRaysOut));
    RAFI_CUDA_CALL_NOTHROW(Free(pDestRank));
    RAFI_CUDA_CALL_NOTHROW(Free(pDestRayID));
  }
  
  template<typename ray_t>
  void RafiImpl<ray_t>::resizeRayQueues(size_t newSize)
  {
    SetActiveGPU forDuration(gpuID);
    
    RAFI_CUDA_CALL(Free(pRaysIn));
    pRaysIn = 0;
    RAFI_CUDA_CALL(Free(pRaysOut));
    pRaysOut = 0;
    RAFI_CUDA_CALL(Free(pDestRank));
    pDestRank = 0;
    RAFI_CUDA_CALL(Free(pDestRayID));
    pDestRayID = 0;

    numReserved = newSize;
    RAFI_CUDA_CALL(Malloc((void **)&pRaysIn,newSize*sizeof(*pRaysIn)));
    RAFI_CUDA_CALL(Malloc((void **)&pRaysOut,newSize*sizeof(*pRaysOut)));
    RAFI_CUDA_CALL(Malloc((void **)&pDestRank,newSize*sizeof(*pDestRank)));
    RAFI_CUDA_CALL(Malloc((void **)&pDestRayID,newSize*sizeof(*pDestRayID)));
  }

  template<typename ray_t>
  DeviceInterface<ray_t> RafiImpl<ray_t>::getDeviceInterface()
  {
    DeviceInterface<ray_t> dd;
    dd.mpi.rank = mpi.rank;
    dd.mpi.size = mpi.size;

    dd.m_numIncoming = numIncoming;
    
    /*! pointer to atomic int for counting outgoing rays */
    dd.pNumOutgoing = pNumOutgoing;
    
    /*! max number of rays reserved for outgoing ray. asking for more
      is a error and will return null */
    dd.maxNumOutgoing = numReserved;

    dd.pDestRayID = pDestRayID;
    dd.pDestRank = pDestRank;
    dd.pRaysOut = pRaysOut;
    dd.pRaysIn = pRaysIn;
    
    return dd;
  }

  template<typename ray_t>
  __global__
  void rearrangeRays(ray_t *pRaysOut,
                     ray_t *pRaysIn, 
                     unsigned *pDestRank,
                     unsigned *pDestRayID,
                     int numRays,
                     int rank)
  {
    int tid = threadIdx.x+blockIdx.x*blockDim.x;
    if (tid >= numRays) return;
    int rid = pDestRayID[tid];
    ray_t rayIn = pRaysIn[rid];
    pRaysOut[tid] = rayIn;
  }

  __global__
  inline void findBegin(int  *d_begin,
                        unsigned *pDestRank,
                        unsigned *pDestRayID,
                        int numRays)
  {
    int tid = threadIdx.x+blockIdx.x*blockDim.x;
    if (tid >= numRays) return;

    if (tid == 0 ||
        pDestRank[tid] != pDestRank[tid-1]) {
      d_begin[pDestRank[tid]] = tid;
    }
  }
  
  template<typename ray_t>
  ForwardResult RafiImpl<ray_t>::forwardRays()
  {
    SetActiveGPU forDuration(gpuID);
    int numOutgoing = 0;
    RAFI_CUDA_CALL(Memcpy(&numOutgoing,pNumOutgoing,sizeof(int),cudaMemcpyDefault));
    if (numOutgoing > 0) {
      unsigned  *d_keys_sorted = 0;
      unsigned  *d_values_sorted = 0;
      RAFI_CUDA_CALL(Malloc((void **)&d_keys_sorted,numOutgoing*sizeof(unsigned)));
      RAFI_CUDA_CALL(Malloc((void **)&d_values_sorted,numOutgoing*sizeof(unsigned)));
      // Determine temporary device storage requirements
      void     *d_temp_storage = nullptr;
      size_t   temp_storage_bytes = 0;
      cub::DeviceRadixSort::SortPairs(d_temp_storage,
                                      temp_storage_bytes,
                                      pDestRank,
                                      d_keys_sorted,
                                      pDestRayID,
                                      d_values_sorted,
                                      (size_t)numOutgoing);
      
      // Allocate temporary storage
      RAFI_CUDA_CALL(Malloc(&d_temp_storage, temp_storage_bytes));
      
      cub::DeviceRadixSort::SortPairs(d_temp_storage,
                                      temp_storage_bytes,
                                      pDestRank,
                                      d_keys_sorted,
                                      pDestRayID,
                                      d_values_sorted,
                                      (size_t)numOutgoing);
      RAFI_CUDA_CALL(Memcpy(pDestRayID,d_values_sorted,
                            numOutgoing*sizeof(unsigned),
                            cudaMemcpyDefault));
      RAFI_CUDA_CALL(Memcpy(pDestRank,d_keys_sorted,
                            numOutgoing*sizeof(unsigned),
                            cudaMemcpyDefault));
      RAFI_CUDA_SYNC_CHECK();
      RAFI_CUDA_CALL(Free(d_keys_sorted));
      RAFI_CUDA_CALL(Free(d_values_sorted));
      RAFI_CUDA_CALL(Free(d_temp_storage));
      // ------------------------------------------------------------------
      // re-arrange rays
      // ------------------------------------------------------------------
      {
        std::swap(pRaysOut,pRaysIn);
        int bs = 1024;
        int nb = divRoundUp(numOutgoing,bs);
        if (nb)
          rearrangeRays<<<nb,bs>>>(pRaysOut,pRaysIn,pDestRank,pDestRayID,
                                   numOutgoing,mpi.rank);
    
      }
    }
    
    // ------------------------------------------------------------------
    // find where ray's offsets are, and use that to compute the
    // per-rank counts
    // ------------------------------------------------------------------
    std::vector<int> begin(mpi.size);
    int *d_begin = 0;
    RAFI_CUDA_CALL(Malloc((void **)&d_begin, mpi.size*sizeof(int)));
    RAFI_CUDA_CALL(Memset((void *)d_begin, -1, mpi.size*sizeof(int)));
    {
      int bs = 1024;
      int nb = divRoundUp(numOutgoing,bs);
      if (nb)
        findBegin<<<nb,bs>>>(d_begin,pDestRank,pDestRayID,numOutgoing);
      RAFI_CUDA_CALL(Memcpy(begin.data(),d_begin,mpi.size*sizeof(int),
                            cudaMemcpyDefault));
      RAFI_CUDA_SYNC_CHECK();
    }
    std::vector<int> end(mpi.size);
    {
      int curEnd = numOutgoing;
      for (int i=mpi.size-1;i>=0;--i) {
        end[i] = curEnd;
        if (begin[i] != -1)
          curEnd = begin[i];
      }
    }
    {
      int curBegin = 0;
      for (int i=0;i<mpi.size;i++) {
        begin[i] = curBegin;
        curBegin = end[i];
      }
    }
    std::vector<int> count(mpi.size);
    for (int i=0;i<mpi.size;i++)
      count[i] = end[i] - begin[i];

    // ------------------------------------------------------------------
    // exchange ray counts
    // ------------------------------------------------------------------
    const std::vector<int> &numRaysWeAreSendingTo = count;
    std::vector<int> numRaysWeAreReceivingFrom(mpi.size);
    RAFI_MPI_CALL(Alltoall(numRaysWeAreSendingTo.data(),1,MPI_INT,
                           numRaysWeAreReceivingFrom.data(),1,MPI_INT,
                           mpi.comm));
    
#if 1
    std::stringstream ss;
    std::vector<MPI_Request> requests;
    ray_t *recvPtr = pRaysIn;
    for (int i=0;i<mpi.size;i++) {
      MPI_Request r;
      int count = numRaysWeAreReceivingFrom[i];
      if (count == 0) continue;
      RAFI_MPI_CALL(Irecv(recvPtr,count*sizeof(ray_t),
                          MPI_BYTE,i,0,mpi.comm,&r));
      requests.push_back(r);
      recvPtr += count;
    }
    numIncoming = recvPtr-pRaysIn;
    
    ray_t *sendPtr = pRaysOut;
    for (int i=0;i<mpi.size;i++) {
      MPI_Request r;
      int count = numRaysWeAreSendingTo[i];
      if (count == 0) continue;
      RAFI_MPI_CALL(Isend(sendPtr,count*sizeof(ray_t),
                          MPI_BYTE,i,0,mpi.comm,&r));
      requests.push_back(r);
      sendPtr += count;
    }
    assert(numOutgoing == sendPtr - pRaysOut);
    RAFI_MPI_CALL(Waitall(requests.size(),requests.data(),MPI_STATUSES_IGNORE));
#else
    // ------------------------------------------------------------------
    // exchange rays themselves
    // ------------------------------------------------------------------
    std::vector<int> recvCounts(mpi.size);
    std::vector<int> recvOffsets(mpi.size);
    std::vector<int> sendCounts(mpi.size);
    std::vector<int> sendOffsets(mpi.size);

    int sendSum = 0;
    for (int i=0;i<mpi.size;i++) {
      sendCounts[i] = numRaysWeAreSendingTo[i]*sizeof(ray_t);
      sendOffsets[i] = sendSum*sizeof(ray_t);
      sendSum += numRaysWeAreSendingTo[i];
    }
    
    int recvSum = 0;
    for (int i=0;i<mpi.size;i++) {
      recvCounts[i] = numRaysWeAreReceivingFrom[i]*sizeof(ray_t);
      recvOffsets[i] = recvSum*sizeof(ray_t);
      recvSum += numRaysWeAreReceivingFrom[i];
    }
    cudaMemset(pRaysIn,-1,recvSum*sizeof(ray_t));
    RAFI_MPI_CALL(Alltoallv(pRaysOut,sendCounts.data(),sendOffsets.data(),MPI_BYTE,
                            pRaysIn,recvCounts.data(),recvOffsets.data(),MPI_BYTE,
                            mpi.comm));

    for (int i=0;i<mpi.size;i++) {
      if (recvCounts[i]) {
        int bs = 1024;
        int nb = divRoundUp(recvCounts[i],bs);
        if (nb)
          checkRays<<<nb,bs>>>(pRaysIn+recvOffsets[i]/sizeof(ray_t),i,mpi.rank,
                               recvCounts[i]/sizeof(ray_t));
      }
    }
    
    numIncoming = recvSum;
#endif

    // ------------------------------------------------------------------
    // cleanup
    // ------------------------------------------------------------------
    RAFI_CUDA_CALL(Memset(pNumOutgoing,0,sizeof(int)));
    RAFI_CUDA_CALL(Free(d_begin));

    ForwardResult result;
    result.numRaysInIncomingQueueThisRank = 0;
    
    for (int i=0;i<mpi.size;i++)
      result.numRaysInIncomingQueueThisRank += numRaysWeAreReceivingFrom[i];
    RAFI_MPI_CALL(Allreduce(&result.numRaysInIncomingQueueThisRank,
                            &result.numRaysAliveAcrossAllRanks,
                            1,MPI_INT,MPI_SUM,mpi.comm));
    assert(this->numIncoming == result.numRaysInIncomingQueueThisRank);
    return result;
  }
  
}

#define RAFI_INSTANTIATE(MyRayT)                                        \
  template rafi::HostContext<MyRayT> *                                  \
  rafi::createContext<MyRayT>(MPI_Comm comm,                            \
                              int gpuID);



