#include <mpi.h>

#define RAFI_MPI_CALL(fctCall)                                          \
  { int rc = MPI_##fctCall;                                             \
    if (rc != MPI_SUCCESS)                                              \
      throw std::runtime_error                                          \
        ("#rafi.mpi (@"+std::string(__PRETTY_FUNCTION__)+") : "         \
         + rafi::mpiErrorString(rc));                                   \
  }
    
namespace rafi {
  
  inline std::string mpiErrorString(int rc)
  {
    char s[MPI_MAX_ERROR_STRING];
    memset(s,0,MPI_MAX_ERROR_STRING);
    int len = MPI_MAX_ERROR_STRING;
    MPI_Error_string(rc,s,&len);
    return s;
  }
}
  
