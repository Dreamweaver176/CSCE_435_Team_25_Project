#!/bin/bash

# for i in {3 5 9 17 33 65 129 257 513 1025}
for i in {513 1025}
do
    for j in {65536 262144 1048576 4194304 16777216 67108864 268435456}
    do
        sbatch mpi.grace_job $j $i s
    done
done

# for i in {3 5 9 17 33 65 129 257 513 1025}
for i in {513 1025}
do
    for j in {65536 262144 1048576 4194304 16777216 67108864 268435456}
    do
        sbatch mpi.grace_job $j $i r
    done
done

# for i in {3 5 9 17 33 65 129 257 513 1025}
for i in {513 1025}
do
    for j in {65536 262144 1048576 4194304 16777216 67108864 268435456}
    do
        sbatch mpi.grace_job $j $i p
    done
done

# for i in {3 5 9 17 33 65 129 257 513 1025}
for i in {513 1025}
do
    for j in {65536 262144 1048576 4194304 16777216 67108864 268435456}
    do
        sbatch mpi.grace_job $j $i a
    done
done