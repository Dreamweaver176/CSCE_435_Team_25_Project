#!/bin/bash

for j in {65536 262144 1048576 4194304 16777216 67108864 268435456}
do
    sbatch mergesort.grace_job $j 64 s
done

for j in {65536 262144 1048576 4194304 16777216 67108864 268435456}
do
    sbatch mergesort.grace_job $j 64 r
done

for j in {65536 262144 1048576 4194304 16777216 67108864 268435456}
do
    sbatch mergesort.grace_job $j 64 p
done

for j in {65536 262144 1048576 4194304 16777216 67108864 268435456}
do
    sbatch mergesort.grace_job $j 64 a
done