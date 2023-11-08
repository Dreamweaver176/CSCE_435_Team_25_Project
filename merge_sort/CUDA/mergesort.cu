#include <stdlib.h>
#include <stdio.h>
#include <time.h>

#include <caliper/cali.h>
#include <caliper/cali-manager.h>
#include <adiak.hpp>

#include <cuda.h>
#include <cuda_runtime.h>

const char* comp = "comp";
const char* comp_large = "comp_large";
const char* merge = "merge";
const char* comp_small = "comp_small";
const char* sequential_sort = "sequential_sort";
const char* whole_computation = "whole_computation";
const char* data_init = "data_init";
const char* transfer_host_to_device = "transfer_host_to_device";
const char* transfer_device_to_host = "transfer_device_to_host";
const char* correctness_check = "correctness_check";


int NUMTHREADS;
int sizeOfMatrix;
int BLOCKS;

__device__ void sequential_sort(int* data, int size) {
    for (int i = 1; i < size; i++) {
        int key = data[i];
        int j = i - 1;
        while (j >= 0 && data[j] > key) {
            data[j + 1] = data[j];
            j = j - 1;
        }
        data[j + 1] = key;
    }
}

__device__ void merge(int* data, int start, int middle, int end) {
    int i = start;
    int j = middle;
    int* tempData = data;
    for(int k = start; k < end; k++) {
        if(i < middle && (j>=end || data[i] < data[j])) {
            data[k] = tempData[j];
            i++;
        }
        else {
            data[k] = tempData[i];
            j++;
        }
    }
    cudaFree(tempData);
}

__device__ void recursive_merge(double *data, int size, int aggregate) {
    int taskid = threadIdx.x;
    if(taskid%aggregate == 1) {
        //sequential merge with start = taskid * (size / blockDim.x), mid = (taskid + 1) * (size / blockDim.x), end = (taskid + (aggregate/2)) * (size / blockDim.x)
        int start = taskid * (size / blockDim.x);
        int mid = (taskid + 1) * (size / blockDim.x);
        int end = (taskid + (aggregate/2)) * (size / blockDim.x);
        CALI_MARK_BEGIN(comp);
        CALI_MARK_BEGIN(comp_large);
        CALI_MARK_BEGIN(merge);
        merge(data, start, mid, end);
        CALI_MARK_END(comp);
        CALI_MARK_END(comp_large);
        CALI_MARK_END(merge);
        
        //recursively merge up
        if((taskid + (aggregate/2)) * (size / blockDim.x) != size) {
            recursive_merge(data, size, aggregate*2);
        }
    }
}

// Define the CUDA kernel for performing merge sort on each block
__global__ void mergeSortKernel(double *data, int size) {
    //figure out which section of the shared, block-level data we're working with
    CALI_CXX_MARK_FUNCTION;
    int taskid = threadIdx.x;
    int start = taskid * (size / blockDim.x);
    int end = (taskid + 1) * (size / blockDim.x);

    //sort data in our little subsection
    CALI_MARK_BEGIN(comp);
    CALI_MARK_BEGIN(comp_small);
    CALI_MARK_BEGIN(sequential_sort);
    sequential_sort(data[start], end-start);
    CALI_MARK_END(comp);
    CALI_MARK_END(comp_small);
    CALI_MARK_END(sequential_sort);

    //recursive merge up
    recursive_merge(data, size, 2);
}

// Define the CUDA kernel for merging sorted blocks, should be performed on a single block theoretically with threads = num_blocks
__global__ void mergeKernel(double *data, int size) {
    //recursive merge up
    recursive_merge(data, size, 2);
}

bool test_array_is_in_order(int arr[]) {
    for (int i = 1; i < LENGTH; i ++) {
        if (arr[i] < arr[i - 1]) {
            printf("Error. Out of order sequence: %d found\n", arr[i]);
            return false;
        }
    }
    printf("Array is in sorted order\n");
    return true;
};

int main() {
    CALI_CXX_MARK_FUNCTION;

    CALI_MARK_BEGIN(whole_computation);

    cali::ConfigManager mgr;
    mgr.start();
 
    NUMTHREADS = atoi(argv[1]);
    sizeOfMatrix = atoi(argv[2]);
    BLOCKS = sizeOfMatrix / NUMTHREADS;

    double[] localArr = new double[sizeOfMatrix];
    double* remoteArr;
    cudaMalloc((void**)&remoteArr, sizeOfMatrix);

    CALI_MARK_BEGIN(data_init);
    for(int i = 0; i < sizeOfMatrix; i++) {
        localArr[i] = srand(1);
    }
    CALI_MARK_END(data_init);

    CALI_MARK_BEGIN(comm);
    CALI_MARK_BEGIN(comm_large);
    CALI_MARK_BEGIN(transfer_host_to_device);
    cudaMemcpy(remoteArr, localArr, sizeOfMatrix, cudaMemcpyHostToDevice);
    CALI_MARK_END(comm);
    CALI_MARK_END(comm_large);
    CALI_MARK_END(transfer_host_to_device);

    dim3 blocks(BLOCKS-1,1);    /* Number of blocks   */
    dim3 threads(NUMTHREADS,1);  /* Number of threads  */
    dim3 kernelBlock(1,1); 

    // Launch the CUDA kernel for performing merge sort on each block
    mergeSortKernel<<<blocks,threads>>>(remoteArr,sizeOfMatrix);

    // Launch the CUDA kernel for merging the sorted blocks
    mergeKernel<<<kernelBlock,threads>>>(remoteArr,sizeOfMatrix);

    // Copy the sorted data back from the GPU to the host
    CALI_MARK_BEGIN(comm);
    CALI_MARK_BEGIN(comm_large);
    CALI_MARK_BEGIN(transfer_device_to_host);
    cudaMemcpy(localArr, remoteArr, sizeOfMatrix, cudaMemcpyDeviceToHost);
    CALI_MARK_END(comm);
    CALI_MARK_END(comm_large);
    CALI_MARK_END(transfer_device_to_host);

    // Free memory on the GPU
    cudaFree(remoteArr);

    // Print the sorted data
    CALI_MARK_BEGIN(correctness_check)
    test_array_is_in_order(localArr);
    CALI_MARK_END(correctness_check)

    CALI_MARK_END(whole_computation);
    

    adiak::init(NULL);
    adiak::launchdate();    // launch date of the job
    adiak::libraries();     // Libraries used
    adiak::cmdline();       // Command line used to launch the job
    adiak::clustername();   // Name of the cluster
    adiak::value("Algorithm", "MergeSort"); // The name of the algorithm you are using (e.g., "MergeSort", "BitonicSort")
    adiak::value("ProgrammingModel", "CUDA"); // e.g., "MPI", "CUDA", "MPIwithCUDA"
    adiak::value("Datatype", "Double"); // The datatype of input elements (e.g., double, int, float)
    adiak::value("SizeOfDatatype", sizeof(Double)); // sizeof(datatype) of input elements in bytes (e.g., 1, 2, 4)
    adiak::value("InputSize", sizeOfMatrix); // The number of elements in input dataset (1000)
    adiak::value("InputType", "Random"); // For sorting, this would be "Sorted", "ReverseSorted", "Random", "1%perturbed"
    adiak::value("num_procs", num_tasks); // The number of processors (MPI ranks)
    adiak::value("num_threads", threads); // The number of CUDA or OpenMP threads
    adiak::value("num_blocks", blocks+kernelBlock); // The number of CUDA blocks 
    adiak::value("group_num", 25); // The number of your group (integer, e.g., 1, 10)
    adiak::value("implementation_source", "Online/AI/Handwritten") // Where you got the source code of your algorithm; choices: ("Online", "AI", "Handwritten").

    return 0;
}
