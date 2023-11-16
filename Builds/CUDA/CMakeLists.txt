cmake_minimum_required(VERSION 3.12)

project(odd_even LANGUAGES CUDA CXX)

find_package(MPI REQUIRED)
find_package(caliper REQUIRED)
find_package(adiak REQUIRED)

add_executable(odd_even odd_even.cu)

message(STATUS "MPI includes : ${MPI_INCLUDE_PATH}")
message(STATUS "CUDA includes : ${CUDA_INCLUDE_DIRS}")
message(STATUS "Caliper includes : ${caliper_INCLUDE_DIR}")
message(STATUS "Adiak includes : ${adiak_INCLUDE_DIRS}")
include_directories(SYSTEM ${MPI_INCLUDE_PATH})
include_directories(${CUDA_INCLUDE_DIRS})
include_directories(${caliper_INCLUDE_DIR})
include_directories(${adiak_INCLUDE_DIRS})

target_link_libraries(odd_even PRIVATE MPI::MPI_CXX)
target_link_libraries(odd_even PRIVATE caliper)