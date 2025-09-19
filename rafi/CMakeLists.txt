find_package(MPI REQUIRED)

add_library(rafi INTERFACE)
target_link_libraries(rafi INTERFACE owl::owl MPI::MPI_CXX)
target_include_directories(rafi INTERFACE
  ${CMAKE_CURRENT_LIST_DIR}/..
)


