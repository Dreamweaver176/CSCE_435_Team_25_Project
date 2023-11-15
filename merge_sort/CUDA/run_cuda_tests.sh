#!/bin/bash

for i in {2 4 8 16 32 64 128}
do
    for j in {128 256 512 1024 4096}
    do
        sbatch mergesort.grace_job $j $i s
    done
done

for i in {2 4 8 16 32 64 128}
do
    for j in {128 256 512 1024 4096}
    do
        sbatch mergesort.grace_job $j $i r
    done
done

for i in {2 4 8 16 32 64 128}
do
    for j in {128 256 512 1024 4096}
    do
        sbatch mergesort.grace_job $j $i p
    done
done

for i in {2 4 8 16 32 64 128}
do
    for j in {128 256 512 1024 4096}
    do
        sbatch mergesort.grace_job $j $i a
    done
done