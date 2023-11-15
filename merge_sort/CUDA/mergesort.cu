#include <stdlib.h>
#include <stdio.h>
#include <time.h>

#include <iostream>
#include <algorithm>
#include <thrust/device_vector.h>                           // ADDED
#include <thrust/copy.h>
#include <thrust/sort.h>

#include <caliper/cali.h>
#include <caliper/cali-manager.h>
#include <adiak.hpp>

#include <cuda_runtime.h>
#include <cuda.h>

const char* comm = "comm";
const char* comm_small = "comm_small";
const char* comm_large = "comm_large";
const char* comp = "comp";
const char* comp_large = "comp_large";
const char* merge_region = "merge";
const char* comp_small = "comp_small";
const char* sequential_sort_region = "sequential_sort";
const char* whole_computation = "whole_computation";
const char* data_init = "data_init";
const char* correctness_check = "correctness_check";
const char* merge_inside_blocks = "merge_inside_blocks";
const char* merge_blocks = "merge_blocks";
const char* transfer_host_to_device = "transfer_host_to_device";
const char* transfer_device_to_host = "transfer_device_to_host";


int NUMTHREADS;
int sizeOfMatrix;
char data_order;
int BLOCKS;

__device__ void sequential_sort(double* data, int start, int end) {
    for (int i = start; i < end; i++) {
        int key = data[i];
        int j = i - 1;
        while (j >= 0 && data[j] > key) {
            data[j + 1] = data[j];
            j = j - 1;
        }
        data[j + 1] = key;
    }
}

__device__ void merge(double *arr, int start, int middle, int end) {
    int i = start;
    int j = middle;
    double *workArr = (double*)malloc((end-start)*sizeof(double));
    memcpy(workArr,arr,(end-start)*sizeof(double));
    for(int k = start; k < end; k++) {
        if(i < middle && (j>=end || workArr[i] < workArr[j])) {
            arr[k] = workArr[i];
            i++;
        }
        else {
            arr[k] = workArr[j];
            j++;
        }
    }
    free(workArr);
}

__device__ void recursive_merge(double *data, int size, int aggregate) {
    int taskid = threadIdx.x;
    if(taskid%aggregate == 1) {
        //sequential merge with start = taskid * (size / blockDim.x), mid = (taskid + 1) * (size / blockDim.x), end = (taskid + (aggregate/2)) * (size / blockDim.x)
        int start = taskid * (size / blockDim.x);
        int mid = (taskid + 1) * (size / blockDim.x);
        int end = (taskid + (aggregate/2)) * (size / blockDim.x);
        
        merge(data, start, mid, end);

        __syncthreads();
        
        //recursively merge up
        if((taskid + (aggregate/2)) * (size / blockDim.x) != size) {
            recursive_merge(data, size, aggregate*2);
        }
    }
}

// Define the CUDA kernel for performing merge sort on each block
__global__ void mergeSortKernel(double *data, int size) {
    //figure out which section of the shared, block-level data we're working with
    int taskid = threadIdx.x;
    int blocks = (size + blockDim.x - 1) / blockDim.x;
    int start = taskid * blocks;
    int end = (taskid + 1) * blocks;
    printf("blockid: %d, taskid: %d, start: %d, end: %d\n", blockIdx.x, taskid, start, end);
    // for(int i = start; i < end; i++) {
    //     printf("taskid: %d - b at %d: %f\n", taskid, i, data[i]);
    // }

    //sort data in our little subsection
    sequential_sort(data, start, end);

    __syncthreads();

    //recursive merge up
    recursive_merge(data, size, 2);
}

// Define the CUDA kernel for merging sorted blocks, should be performed on a single block theoretically with threads = num_blocks
__global__ void mergeKernel(double *data, int size) {
    //recursive merge up
    recursive_merge(data, size, 2);
}

void fill_data(double *arr, char data_order, int sizeOfMatrix) {
    if(data_order == 's') {
        for(int i = 0; i < sizeOfMatrix; i++) {
            arr[i] = (double)i;
        }
    }
    else if(data_order == 'r') {
        int j = 0;
        for(int i = sizeOfMatrix; i > 0; i--) {
            arr[j] = (double)i;
            j++;
        }
    }
    else if(data_order == 'p') {
        srand(187);
        int perturbed_offset = sizeOfMatrix/100;
        for(int i = 0; i < sizeOfMatrix; i++) {
            arr[i] = (double)i;
        }
        for(int i = 0; i < perturbed_offset; i++) {
            int add1 = rand()%sizeOfMatrix;
            int add2 = rand()%sizeOfMatrix;
            double temp = arr[add1];
            arr[add1] = arr[add2];
            arr[add2] = temp;
        }
    }
    else if(data_order == 'a') {
        srand(187);
        for(int i = 0; i < sizeOfMatrix; i++) {
            arr[i] = rand();
        }
    }
}

int main(int argc, char *argv[]) {
    CALI_CXX_MARK_FUNCTION;

    // Create caliper ConfigManager object
    cali::ConfigManager mgr;
    mgr.start();
    
    srand(187);
    sizeOfMatrix = atoi(argv[1]);
    NUMTHREADS = atoi(argv[2]);
    data_order = *(argv[3]);
    BLOCKS = (sizeOfMatrix + NUMTHREADS - 1) / NUMTHREADS;

    double *localArr = (double*)malloc(sizeOfMatrix * sizeof(double));
    double* remoteArr;
    cudaMalloc((void**)&remoteArr, sizeOfMatrix*sizeof(double));

    CALI_MARK_BEGIN(data_init);
    fill_data(localArr, data_order, sizeOfMatrix);
    CALI_MARK_END(data_init);

    CALI_MARK_BEGIN(comm);
    CALI_MARK_BEGIN(comm_large);
    cudaMemcpy(remoteArr, localArr, sizeOfMatrix, cudaMemcpyHostToDevice);
    CALI_MARK_END(comm_large);
    CALI_MARK_END(comm);

    dim3 blocks(BLOCKS,1);    /* Number of blocks   */
    dim3 threads(NUMTHREADS,1);  /* Number of threads  */
    dim3 kernelBlock(1,1);

    // Launch the CUDA kernel for performing merge sort on each block
    CALI_MARK_BEGIN(comp);
    CALI_MARK_BEGIN(comp_large);
    mergeSortKernel<<<BLOCKS,threads>>>(remoteArr,sizeOfMatrix);
    CALI_MARK_END(comp_large);
    CALI_MARK_END(comp);

    

    // Launch the CUDA kernel for merging the sorted blocks
    CALI_MARK_BEGIN(comp);
    CALI_MARK_BEGIN(comp_large);
    mergeKernel<<<kernelBlock,threads>>>(remoteArr,sizeOfMatrix);
    CALI_MARK_END(comp_large);
    CALI_MARK_END(comp);

    // Copy the sorted data back from the GPU to the host
    CALI_MARK_BEGIN(comm);
    CALI_MARK_BEGIN(comm_large);
    cudaMemcpy(localArr, remoteArr, sizeOfMatrix, cudaMemcpyDeviceToHost);
    CALI_MARK_END(comm_large);
    CALI_MARK_END(comm);
    
    // Free memory on the GPU
    cudaFree(remoteArr);

    // Print the sorted data
    bool err_found = false;
    CALI_MARK_BEGIN(correctness_check);
    for (int i = 1; i < sizeOfMatrix; i++) {
            if (localArr[i] < localArr[i - 1]) {
                printf("Error. Out of order sequence: %d found at: %d after value: %d\n", localArr[i], i, localArr[i-1]);
                err_found = true;
                break;
            }
        }
        if(err_found) {
            for(int i = 0; i < sizeOfMatrix; i++) {
                // printf("b at %d: %f\n", i, localArr[i]);
            }
        }
        else {
            printf("Array is properly sorted\n");
        }
    CALI_MARK_END(correctness_check);
    free(localArr);

    adiak::init(NULL);
    adiak::launchdate();    // launch date of the job
    adiak::libraries();     // Libraries used
    adiak::cmdline();       // Command line used to launch the job
    adiak::clustername();   // Name of the cluster
    adiak::value("Algorithm", "MergeSort"); // The name of the algorithm you are using (e.g., "MergeSort", "BitonicSort")
    adiak::value("ProgrammingModel", "CUDA"); // e.g., "MPI", "CUDA", "MPIwithCUDA"
    adiak::value("Datatype", "Double"); // The datatype of input elements (e.g., double, int, float)
    adiak::value("SizeOfDatatype", sizeof(double)); // sizeof(datatype) of input elements in bytes (e.g., 1, 2, 4)
    adiak::value("InputSize", sizeOfMatrix); // The number of elements in input dataset (1000)
    if(data_order == 's') {
        adiak::value("InputType", "Sorted"); // For sorting, this would be "Sorted", "ReverseSorted", "Random", "1%perturbed"
    }
    else if(data_order == 'r') {
        adiak::value("InputType", "ReverseSorted"); // For sorting, this would be "Sorted", "ReverseSorted", "Random", "1%perturbed"
    }
    else if(data_order == 'p') {
        adiak::value("InputType", "1%perturbed"); // For sorting, this would be "Sorted", "ReverseSorted", "Random", "1%perturbed"
    }
    else if(data_order == 'a') {
        adiak::value("InputType", "Random"); // For sorting, this would be "Sorted", "ReverseSorted", "Random", "1%perturbed"
    }    adiak::value("num_procs", 1); // The number of processors (MPI ranks)
    adiak::value("num_threads", NUMTHREADS); // The number of CUDA or OpenMP threads
    adiak::value("num_blocks", BLOCKS+1); // The number of CUDA blocks 
    adiak::value("group_num", 25); // The number of your group (integer, e.g., 1, 10)
    adiak::value("implementation_source", "Online/AI/Handwritten"); // Where you got the source code of your algorithm; choices: ("Online", "AI", "Handwritten").

    // Flush Caliper output before finalizing MPI
    mgr.stop();
    mgr.flush();
};
