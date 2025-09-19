#include <cuda_runtime.h>

namespace rafi {
inline void rafiRaise_impl(std::string str)
{
  fprintf(stderr,"%s\n",str.c_str());
#ifdef WIN32
  if (IsDebuggerPresent())
    DebugBreak();
  else
    throw std::runtime_error(str);
#else
#ifndef NDEBUG
  std::string bt = ::detail::backtrace();
  fprintf(stderr,"%s\n",bt.c_str());
#endif
  raise(SIGINT);
#endif
}
}

#define RAFI_RAISE(MSG) ::rafi::rafiRaise_impl(MSG);


#define RAFI_CUDA_CHECK( call )                                          \
  {                                                                     \
    cudaError_t rc = call;                                              \
    if (rc != cudaSuccess) {                                            \
      fprintf(stderr,                                                   \
              "CUDA call (%s) failed with code %d (line %d): %s\n",     \
              #call, rc, __LINE__, cudaGetErrorString(rc));             \
      RAFI_RAISE("fatal cuda error");                                    \
    }                                                                   \
  }

#define RAFI_CUDA_CALL(call) RAFI_CUDA_CHECK(cuda##call)

#define RAFI_CUDA_CHECK2( where, call )                                  \
  {                                                                     \
    cudaError_t rc = call;                                              \
    if(rc != cudaSuccess) {                                             \
      if (where)                                                        \
        fprintf(stderr, "at %s: CUDA call (%s) "                        \
                "failed with code %d (line %d): %s\n",                  \
                where,#call, rc, __LINE__, cudaGetErrorString(rc));     \
      fprintf(stderr,                                                   \
              "CUDA call (%s) failed with code %d (line %d): %s\n",     \
              #call, rc, __LINE__, cudaGetErrorString(rc));             \
      RAFI_RAISE("fatal cuda error");                                    \
    }                                                                   \
  }

#define RAFI_CUDA_SYNC_CHECK()                                           \
  {                                                                     \
    cudaError_t rc = cudaDeviceSynchronize();                           \
    if (rc != cudaSuccess) {                                            \
      fprintf(stderr, "error (%s: line %d): %s\n",                      \
              __FILE__, __LINE__, cudaGetErrorString(rc));              \
      RAFI_RAISE("fatal cuda error");                                    \
    }                                                                   \
  }

#define RAFI_CUDA_SYNC_CHECK_STREAM(s)                           \
  {                                                             \
    cudaError_t rc = cudaStreamSynchronize(s);                  \
    if (rc != cudaSuccess) {                                    \
      fprintf(stderr, "error (%s: line %d): %s\n",              \
              __FILE__, __LINE__, cudaGetErrorString(rc));      \
      RAFI_RAISE("fatal cuda error");                            \
    }                                                           \
  }



#define RAFI_CUDA_CHECK_NOTHROW( call )                                  \
  {                                                                     \
    cudaError_t rc = call;                                              \
    if (rc != cudaSuccess) {                                            \
      fprintf(stderr,                                                   \
              "CUDA call (%s) failed with code %d (line %d): %s\n",     \
              #call, rc, __LINE__, cudaGetErrorString(rc));             \
      exit(2);                                                          \
    }                                                                   \
  }

#define RAFI_CUDA_CALL_NOTHROW(call) RAFI_CUDA_CHECK_NOTHROW(cuda##call)

#define RAFI_CUDA_CHECK2_NOTHROW( where, call )                          \
  {                                                                     \
    cudaError_t rc = call;                                              \
    if(rc != cudaSuccess) {                                             \
      if (where)                                                        \
        fprintf(stderr, "at %s: CUDA call (%s) "                        \
                "failed with code %d (line %d): %s\n",                  \
                where,#call, rc, __LINE__, cudaGetErrorString(rc));     \
      fprintf(stderr,                                                   \
              "CUDA call (%s) failed with code %d (line %d): %s\n",     \
              #call, rc, __LINE__, cudaGetErrorString(rc));             \
      exit(2);                                                          \
    }                                                                   \
  }

