#pragma once

#include "rafi/rafi.h"
#include "rafi/cuda_check.h"
#include "rafi/mpi_check.h"
#include <cub/cub.cuh>

namespace rafi {

  inline int divRoundUp(int a, int b) { return (a+b-1)/b; }

  template<typename ray_t>
  __global__ void printDebugRays(int rank, ray_t *rays, int numRays)
  {
    int tid = threadIdx.x+blockIdx.x*blockDim.x;
    if (tid >= numRays) return;

    if (rays[tid].dbg) printf("[%i] !!!!!! debug ray at %i\n",rank,tid);
  }
  
  template<typename ray_t>
  struct RafiImpl : public HostContext<ray_t>
  {
    RafiImpl(MPI_Comm comm);
    ~RafiImpl() override;
    void resizeRayQueues(size_t maxRaysOnAnyRankAtAnyTime) override;
    void clearQueue() override {
      cudaMemset(pNumOutgoing,0,sizeof(int));
    }

    DeviceInterface<ray_t> getDeviceInterface() override;
    ForwardResult forwardRays(const char *msg) override;

    int       numIncoming  = 0;
    int      *pNumOutgoing = 0;
    ray_t    *pRaysIn      = 0;
    ray_t    *pRaysOut     = 0;
    unsigned *pDestRank    = 0;
    unsigned *pDestRayID   = 0;
    int       numReserved  = 0;
    using HostContext<ray_t>::mpi;
  };
    
  template<typename ray_t>
  HostContext<ray_t> *createContext(MPI_Comm comm)
  {
    return new RafiImpl<ray_t>(comm);
  }

  
  template<typename ray_t>
  RafiImpl<ray_t>::RafiImpl(MPI_Comm comm)
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
                     int numRays)
  {
    int tid = threadIdx.x+blockIdx.x*blockDim.x;
    if (tid >= numRays) return;
    int dst = pDestRank[tid];
    int rid = pDestRayID[tid];
    if (dst != 0 && dst != 1) {
      printf("dst %i (%i %f)\n",dst,dst,(const float&)dst);
      return;
    }
    if (rid < 0 || rid >= numRays) {
      printf("rid %i\n",rid);
      return;
    }
    ray_t rayIn = pRaysIn[rid];
    if (rayIn.dbg) printf("rearrange WRITING debug ray at %i\n",tid);
    pRaysOut[tid] = rayIn;
  }

  __global__
  void checkDests(int stage,
                  unsigned *pDestRank,
                  unsigned *pDestRayID,
                  int numRays)
  {
    int tid = threadIdx.x+blockIdx.x*blockDim.x;
    if (tid >= numRays) return;
    int dst = pDestRank[tid];
    int rid = pDestRayID[tid];
    if (numRays <= 32)
      printf("dst<%i>[%i/%i] = %i %i\n",stage,tid,numRays,dst,rid);
    if (dst != 0 && dst != 1) {
      printf("check<%i>[%i/%i] dst %i\n",stage,tid,numRays,dst);
      return;
    }
    if (rid < 0 || rid >= numRays) {
      printf("check<%i>[%i/%i] rid %i\n",stage,tid,numRays,rid);
      return;
    }
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
  ForwardResult RafiImpl<ray_t>::forwardRays(const char *bla)
  {
    ForwardResult result;

    RAFI_CUDA_SYNC_CHECK();
    int numOutgoing = 0;
    RAFI_CUDA_CALL(Memcpy(&numOutgoing,pNumOutgoing,sizeof(int),cudaMemcpyDefault));
    RAFI_CUDA_SYNC_CHECK();

    printf("rafi[%i] forwarding rays count=%i %s\n",mpi.rank,numOutgoing,bla);
    if (numOutgoing > 0)
      printDebugRays<<<divRoundUp(numOutgoing,1024),1024>>>(mpi.rank,pRaysOut,numOutgoing);
    RAFI_CUDA_SYNC_CHECK();
    
    // ------------------------------------------------------------------
    // sort rayID:destRank array
    // ------------------------------------------------------------------
    {
      int bs = 1024;
      int nb = divRoundUp(numOutgoing,bs);
      if (nb)
        checkDests<<<nb,bs>>>(1,pDestRank,pDestRayID,numOutgoing);
      RAFI_CUDA_SYNC_CHECK();
    }

#if 0
    std::vector<uint64_t> hostKeys(numOutgoing);
    cudaMemcpy(hostKeys.data(),
               pDestOut,
               numOutgoing*sizeof(uint64_t),cudaMemcpyDefault);
    RAFI_CUDA_SYNC_CHECK();
    std::sort(hostKeys.begin(),hostKeys.end());
    cudaMemcpy(pDestOut,
               hostKeys.data(),
               numOutgoing*sizeof(uint64_t),cudaMemcpyDefault);
    RAFI_CUDA_SYNC_CHECK();
#else
    // uint64_t  *d_keys_in  = (uint64_t*)pDestOut;
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
    RAFI_CUDA_SYNC_CHECK();
    RAFI_CUDA_CALL(Malloc(&d_temp_storage, temp_storage_bytes));
    RAFI_CUDA_SYNC_CHECK();

    if (numOutgoing > 0) {
    // Run sorting operation
    cudaSetDevice(0);
    cub::DeviceRadixSort::SortPairs(d_temp_storage,
                                    temp_storage_bytes,
                                    pDestRank,
                                    d_keys_sorted,
                                    pDestRayID,
                                    d_values_sorted,
                                    (size_t)numOutgoing);
    RAFI_CUDA_SYNC_CHECK();
    {
      int bs = 1024;
      int nb = divRoundUp(numOutgoing,bs);
      if (nb)
        checkDests<<<nb,bs>>>(2,d_keys_sorted,d_values_sorted,numOutgoing);
      RAFI_CUDA_SYNC_CHECK();
    }
    RAFI_CUDA_SYNC_CHECK();
    
    RAFI_CUDA_SYNC_CHECK();
    RAFI_CUDA_CALL(Memcpy(pDestRayID,d_values_sorted,numOutgoing*sizeof(unsigned),
                          cudaMemcpyDefault));
    RAFI_CUDA_CALL(Memcpy(pDestRank,d_keys_sorted,numOutgoing*sizeof(unsigned),
                          cudaMemcpyDefault));
    RAFI_CUDA_SYNC_CHECK();
    RAFI_CUDA_CALL(Free(d_keys_sorted));
    RAFI_CUDA_CALL(Free(d_values_sorted));
    RAFI_CUDA_CALL(Free(d_temp_storage));
    }
#endif
    RAFI_CUDA_SYNC_CHECK();
    {
      int bs = 1024;
      int nb = divRoundUp(numOutgoing,bs);
      if (nb)
        checkDests<<<nb,bs>>>(3,pDestRank,pDestRayID,numOutgoing);
      RAFI_CUDA_SYNC_CHECK();
    }
    RAFI_CUDA_SYNC_CHECK();

    // ------------------------------------------------------------------
    // re-arrange rays
    // ------------------------------------------------------------------
    {
      int bs = 1024;
      int nb = divRoundUp(numOutgoing,bs);
      if (nb)
        rearrangeRays<<<nb,bs>>>(pRaysIn,pRaysOut,pDestRank,pDestRayID,numOutgoing);
    }
    std::swap(pRaysOut,pRaysIn);
    RAFI_CUDA_SYNC_CHECK();
    printf("rafi[%i] reordered rays for forwarding, count=%i\n",mpi.rank,numOutgoing);
    if (numOutgoing > 0)
      printDebugRays<<<divRoundUp(numOutgoing,1024),1024>>>
        (mpi.rank,pRaysOut,numOutgoing);
    RAFI_CUDA_SYNC_CHECK();
    
    // ------------------------------------------------------------------
    // find where ray's offsets are, and use that to compute the
    // per-rank counts
    // ------------------------------------------------------------------
    std::vector<int> begin(mpi.size);
    int *d_begin = 0;
    RAFI_CUDA_SYNC_CHECK();
    RAFI_CUDA_CALL(Malloc((void **)&d_begin, mpi.size*sizeof(int)));
    RAFI_CUDA_CALL(Memset((void *)d_begin, -1, mpi.size*sizeof(int)));
    RAFI_CUDA_SYNC_CHECK();
    {
      int bs = 1024;
      int nb = divRoundUp(numOutgoing,bs);
      if (nb)
        findBegin<<<nb,bs>>>(d_begin,pDestRank,pDestRayID,numOutgoing);
      RAFI_CUDA_CALL(Memcpy(begin.data(),d_begin,mpi.size*sizeof(int),
                            cudaMemcpyDefault));
      RAFI_CUDA_SYNC_CHECK();
    }
    RAFI_CUDA_SYNC_CHECK();
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
      // printf("send[%i] ofs %i cnt %i\n",i,sendOffsets[i],sendCounts[i]);
    }
    int recvSum = 0;
    for (int i=0;i<mpi.size;i++) {
      recvCounts[i] = numRaysWeAreReceivingFrom[i]*sizeof(ray_t);
      recvOffsets[i] = recvSum*sizeof(ray_t);
      recvSum += numRaysWeAreReceivingFrom[i];
      // printf("recv[%i] ofs %i cnt %i\n",i,recvOffsets[i],recvCounts[i]);
    }
    RAFI_MPI_CALL(Alltoallv(pRaysOut,sendCounts.data(),sendOffsets.data(),MPI_BYTE,
                            pRaysIn,recvCounts.data(),recvOffsets.data(),MPI_BYTE,
                            mpi.comm));
    
    // ------------------------------------------------------------------
    // swap queues
    // ------------------------------------------------------------------
    std::swap(pRaysOut,pRaysIn);
    numIncoming = numOutgoing;
    RAFI_CUDA_CALL(Memset(pNumOutgoing,0,sizeof(int)));

    printf("rafi[%i] doen forwarding, count=%i\n",mpi.rank,numOutgoing);
    if (numOutgoing > 0)
      printDebugRays<<<divRoundUp(numOutgoing,1024),1024>>>(mpi.rank,pRaysOut,numOutgoing);
    RAFI_CUDA_SYNC_CHECK();
    // ------------------------------------------------------------------
    // cleanup
    // ------------------------------------------------------------------
    RAFI_CUDA_CALL(Free(d_begin));
    RAFI_CUDA_SYNC_CHECK();

    result.numRaysInIncomingQueueThisRank = 0;
    for (int i=0;i<mpi.size;i++)
      result.numRaysInIncomingQueueThisRank += numRaysWeAreReceivingFrom[i];
    RAFI_MPI_CALL(Allreduce(&result.numRaysInIncomingQueueThisRank,
                            &result.numRaysAliveAcrossAllRanks,
                            1,MPI_INT,MPI_SUM,mpi.comm));
                  
    return result;
  }
  
}

#define RAFI_INSTANTIATE(MyRayT)                                        \
  template rafi::HostContext<MyRayT> *rafi::createContext<MyRayT>(MPI_Comm comm);



