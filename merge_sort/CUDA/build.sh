#!/bin/bash

module purge

module load CUDA
module load intel/2020b
module load GCCcore/7.3.0
module load CMake/3.12.1

cmake \
    #   -Dcaliper_DIR=/scratch/user/crnicholls20/bin/usr/local/share/cmake/caliper \
    -Dcaliper_DIR=/scratch/group/csce435-f23/Caliper/caliper/share/cmake/caliper \
    -Dadiak_DIR=/scratch/group/csce435-f23/Adiak/adiak/lib/cmake/adiak \
    .

make