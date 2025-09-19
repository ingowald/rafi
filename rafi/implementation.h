#pragma once

#include "rafi/rafi.h"
#include "rafi/cuda_check.h"

namespace rafi {

  template<typename ray_t>
  struct RafiImpl : public HostContext<ray_t>
  {
    RafiImpl(MPI_Comm comm);
    ~RafiImpl() override;
    void resizeRayQueues(size_t maxRaysOnAnyRankAtAnyTime) override;
    DeviceInterface<ray_t> getDeviceInterface() override;
    ForwardResult forwardRays() override;

    int       numIncoming  = 0;
    int      *pNumOutgoing = 0;
    ray_t    *pRaysIn      = 0;
    ray_t    *pRaysOut     = 0;
    int2     *pDestOut     = 0;
    int       numReserved  = 0;
    struct {
      MPI_Comm comm;
      int rank;
      int size;
    } mpi;
    int *d_rankBegin = 0;
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

    RAFI_CUDA_CALL(Malloc((void**)&d_rankBegin,mpi.size*sizeof(int)));
  }
  
  template<typename ray_t>
  RafiImpl<ray_t>::~RafiImpl()
  {
    RAFI_CUDA_CALL_NOTHROW(Free(pNumOutgoing));
    RAFI_CUDA_CALL_NOTHROW(Free(pRaysIn));
    RAFI_CUDA_CALL_NOTHROW(Free(pRaysOut));
    RAFI_CUDA_CALL_NOTHROW(Free(pDestOut));
    RAFI_CUDA_CALL_NOTHROW(Free(d_rankBegin));
  }
  
  template<typename ray_t>
  void RafiImpl<ray_t>::resizeRayQueues(size_t newSize)
  {
    RAFI_CUDA_CALL(Free(pRaysIn));
    pRaysIn = 0;
    RAFI_CUDA_CALL(Free(pRaysOut));
    pRaysOut = 0;
    RAFI_CUDA_CALL(Free(pDestOut));
    pDestOut = 0;

    numReserved = newSize;
    RAFI_CUDA_CALL(Malloc((void **)&pRaysIn,newSize*sizeof(*pRaysIn)));
    RAFI_CUDA_CALL(Malloc((void **)&pRaysOut,newSize*sizeof(*pRaysOut)));
    RAFI_CUDA_CALL(Malloc((void **)&pDestOut,newSize*sizeof(*pDestOut)));
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

    dd.pDestOut = pDestOut;
    dd.pRaysOut = pRaysOut;
    dd.pRaysIn = pRaysIn;
    
    return dd;
  }

  template<typename ray_t>
  ForwardResult RafiImpl<ray_t>::forwardRays()
  {
    ForwardResult result;

    RAFI_CUDA_SYNC_CHECK();
    int numOutgoing = 0;
    RAFI_CUDA_CALL(Memcpy(&numOutgoing,pNumOutgoing,sizeof(int),cudaMemcpyDefault));
    RAFI_CUDA_SYNC_CHECK();

    // ------------------------------------------------------------------
    // sort rayID:destRank array
    // ------------------------------------------------------------------

    int  num_items = numOutgoing;
    uint64_t  *d_keys_in  = (uint64_t*)pDestOut;
    uint64_t  *d_keys_out = 0;
    RAFI_CUDA_CALL(Malloc((void **)&d_keys_out,numOutgoing*sizeof(uint64_t)));
    // Determine temporary device storage requirements
    void     *d_temp_storage = nullptr;
    size_t   temp_storage_bytes = 0;
    cub::DeviceRadixSort::SortKeys(d_temp_storage,
                                   temp_storage_bytes,
                                   d_keys_in,
                                   d_keys_out,
                                   num_items);
    
    // Allocate temporary storage
    RAFI_CUDA_CALL(Malloc(&d_temp_storage, temp_storage_bytes));
    
    // Run sorting operation
    cub::DeviceRadixSort::SortKeys(d_temp_storage,
                                   temp_storage_bytes,
                                   d_keys_in,
                                   d_keys_out,
                                   num_items);

    // ------------------------------------------------------------------
    // re-arrange rays
    // ------------------------------------------------------------------
    {
      int bs = 1024;
      int nb = divRoundUp(numOutgoing,bs);
      rearrangeRays<<<nb,bs>>>(dRaysIn,dRaysOut,d_keys_out,numOutgoing);
    }
    std::swap(dRaysOut,dRaysIn);
    
    // ------------------------------------------------------------------
    // find where ray's offsets are, and use that to compute the
    // per-rank counts
    // ------------------------------------------------------------------
    std::vector<int> begin(mpi.size);
    int *d_begin = 0;
    RAFI_CUDA_CALL(Malloc((void **)&d_begin, mpi.size*sizeof(int)));
    RAFI_CUDA_CALL(Memset((void **)&d_begin, -1, mpi.size*sizeof(int)));
    std::vector<int> end(mpi.size);
    int *d_end = 0;
    RAFI_CUDA_CALL(Malloc((void **)&d_end, mpi.size*sizeof(int)));
    RAFI_CUDA_CALL(Memset((void **)&d_end, -1, mpi.size*sizeof(int)));
    {
      int bs = 1024;
      int nb = divRoundUp(numOutgoing,bs);
      findBegin<<<nb,bs>>>(d_begin,d_end,(int2*)d_keys_out,numOutgoing);
      RAFI_CUDA_CALL(Memcpy(begin.data(),d_begin,mpi.size*sizeof(int),
                            cudaMemcpyDefault));
      RAFI_CUDA_SYNC_CHECK();
    }
    std::vector<int> count(mpi.size);
    for (int i=0;i<mpi.size;i++)
      count[i]
        = (begin[i] == -1)
        ? 0
        : (end[i] - begin[i]);
    
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
    RAFI_MPI_CALL(Alltoall(pRaysOut,sendCounts.data(),sendOffsets.data(),MPI_BYTE,
                           pRaysIn,recvCounts.data(),recvOffsets.data(),MPI_BYTE,
                           mpi.comm));

    // ------------------------------------------------------------------
    // swap queues
    // ------------------------------------------------------------------
    std::swap(pRaysOut,pRaysIn);
    numIncoming = numOutgoing;
    RAFI_CUDA_CALL(Memset(pNumOutgoing,0,sizeof(int)));
    // ------------------------------------------------------------------
    // cleanup
    // ------------------------------------------------------------------
    RAFI_CUDA_CALL(Free(d_begin));
    RAFI_CUDA_CALL(Free(d_end));
    RAFI_CUDA_CALL(Free(d_temp_storage));
    RAFI_CUDA_CALL(Free(d_keys_out));
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



