#!/bin/bash

for i in {3 5 9 17 33 65 129}
do
    for j in {128 256 512 1024 4096}
    do
        sbatch mpi.grace_job $j $i s
    done
done

for i in {3 5 9 17 33 65 129}
do
    for j in {128 256 512 1024 4096}
    do
        sbatch mpi.grace_job $j $i r
    done
done

for i in {3 5 9 17 33 65 129}
do
    for j in {128 256 512 1024 4096}
    do
        sbatch mpi.grace_job $j $i p
    done
done

for i in {3 5 9 17 33 65 129}
do
    for j in {128 256 512 1024 4096}
    do
        sbatch mpi.grace_job $j $i a
    done
done