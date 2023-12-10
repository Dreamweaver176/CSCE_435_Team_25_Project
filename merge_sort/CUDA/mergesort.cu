#include <stdlib.h>
#include <stdio.h>
#include <time.h>

#include <iostream>
#include <algorithm>
#include <cmath>
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

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}


int NUMTHREADS;
int sizeOfMatrix;
char data_order;
int BLOCKS;

// function to swap elements
__device__ void swap(double &a, double &b) {
    double temp = a;
    a = b;
    b = temp;
}

// function to rearrange array (find the partition point)
__device__ int partition(double* arr, int low, int high) {
    double pivot = arr[high];
    int i = low - 1;

    for (int j = low; j < high; j++) {
        if (arr[j] < pivot) {
            i++;
            swap(arr[i], arr[j]);
        }
    }

    swap(arr[i + 1], arr[high]);
    return i + 1;
}

__device__ void quickSort(double *array, int low, int high) {
  if (low < high) {
    int pi = partition(array, low, high);
    
    quickSort(array, low, pi - 1);
    quickSort(array, pi + 1, high);
  }
}

__device__ void deathSort(double *arr, int start, int end)
{
    int i, j, min_idx;
 
    // One by one move boundary of
    // unsorted subarray
    for (i = start; i < end - 1; i++) {
 
        // Find the minimum element in
        // unsorted array
        min_idx = i;
        for (j = i + 1; j < end; j++) {
            if (arr[j] < arr[min_idx])
                min_idx = j;
        }
 
        // Swap the found minimum element
        // with the first element
        if (min_idx != i)
            swap(arr[min_idx], arr[i]);
    }
}

// __device__ void merge(double *arr, int start, int middle, int end) {
//     int i = start;
//     int j = middle;
//     double *workArr = (double*)malloc((end-start)*sizeof(double));
//     memcpy(workArr,arr + i,(end-start)*sizeof(double));
//     // if(blockIdx.x == 2) {
//     //     for(int i = start; i < end; i++) {
//     //         printf("blockid: %d, taskid: %d - b at %d: %f\n", blockIdx.x, threadIdx.x, i, arr[i]);
//     //     }
//     //     printf("\n");
//     // }
//     for(int k = start; k < end; k++) {
//         if(i < middle && (j>=end || workArr[i] < workArr[j])) {
//             arr[k] = workArr[i];
//             i++;
//         }
//         else {
//             arr[k] = workArr[j];
//             j++;
//         }
//         // printf("blockid: %d, taskid: %d - b at %d: %f\n", blockIdx.x, threadIdx.x, k, arr[k]);
//     }
//     free(workArr);
// }

__device__ void mergeTwoSortedArrays(double *data, double *workArr, int start, int mid, int end) {
    int i = start;
    int j = mid;
    int k = start;
    int count = 0;
    if(mid < start || mid > end || start > end) {
        printf("issue with bounds: start: %d mid: %d end: %d\n", start, mid, end);
    }
    // if(threadIdx.x == 8 && blockIdx.x == 4) 
    //     printf("pre memcpy: wa: %f data: %f\n", workArr[k], data[i]);
    // memcpy(workArr+start, data+start, (end-start)*sizeof(double));
    // if(threadIdx.x == 8 && blockIdx.x == 4) 
    //     printf("post memcpy: wa: %f data: %f\n", workArr[k], workArr[i]);
    while(i < mid && j < end) {
        if(data[i] < data[j]) {
            workArr[k] = data[i];
            // if(threadIdx.x == 8 && blockIdx.x == 4) 
            //     printf("j > i: wa: %f data: %f\n", workArr[k], data[i]);
            k++;
            i++;
            count++;
        }
        else {
            // if(threadIdx.x == 8 && blockIdx.x == 4) 
            //     printf("test var: data[j+1]: %f\n", data[j+1]);
            workArr[k] = data[j];
            // if(threadIdx.x == 8 && blockIdx.x == 4) 
            //     printf("i > j: wa: %f data: %f\n", workArr[k], data[j]);
            k++;
            j++;
            count++;
        }
        if(k > end) {
            printf("merge error with k %d\n", k);
        }
    }
    if(i == mid) {
        // if(threadIdx.x == 8 && blockIdx.x == 4) 
        //     printf("i is mid\n");
        while(j < end) {
            workArr[k] = data[j];
            // if(threadIdx.x == 8 && blockIdx.x == 4) 
            //     printf("i==mid: wa: %f data: %f\n", workArr[k], data[j]);
            k++;
            j++;
            count++;
        }
    }
    if(j == end) {
        // if(threadIdx.x == 8 && blockIdx.x == 4) 
        //     printf("j is end\n");
        while(i < mid) {
            workArr[k] = data[i];
            // if(threadIdx.x == 8 && blockIdx.x == 4) 
            //     printf("j==end: wa: %f data: %f\n", workArr[k], data[i]);
            k++;
            i++;
            count++;
        }
    }
    if(k > end) {
        printf("merge error with k %d\n", k);
    }
    // if(threadIdx.x == 8 && blockIdx.x == 4) {
    //     for(int i = start; i < end; i++) {
    //         printf("In Merge: blockid: %d, taskid: %d - b at %d: %f\n", blockIdx.x, threadIdx.x, i, workArr[i]);
    //     }
    // }
    // if(threadIdx.x == 8 && blockIdx.x == 4) 
    //     printf("entry: %f, entry2: %f\n", data[end-1], workArr[end-1]);
    memcpy(data+start, workArr+start, (end-start)*sizeof(double));
}


__device__ void mergeUp(double *data, double *workArr, int size, int aggregate) {
    int taskid = threadIdx.x;
    int threadSize = size / blockDim.x;
    // if(threadIdx.x == 0 && aggregate > 2) {
    //         for(int i = blockstart; i < (blockstart + (blockDim.x * blockDim.x)); i++) {
                // printf("Block %d Thread %d Agg %d - start: %d mid: %d end: %d\n", blockIdx.x, threadIdx.x, i, aggregate, data[i]);
    //         }
    // }
    if(taskid%aggregate == 0) {
        //sequential merge with start = taskid * (size / blockDim.x), mid = (taskid + 1) * (size / blockDim.x), end = (taskid + (aggregate/2)) * (size / blockDim.x)
        int start = (taskid) * threadSize;
        int end = ((taskid + aggregate) * threadSize);
        int mid = ((end - start)/2) + start;
        // for(int i = blockstart; i < (blockstart + (blockDim.x * blockDim.x)); i++) {
        // if(aggregate > 128)
        //     printf("MergeKernel: Block %d Thread %d Agg %d - start: %d mid: %d end: %d\n", blockIdx.x, threadIdx.x, aggregate, start, mid, end);
        // }
        mergeTwoSortedArrays(data, workArr, start, mid, end);
        // if(aggregate > 64)
        //     printf("MergeKernel: Block %d Thread %d Agg %d - start: %d mid: %d end: %d\n", blockIdx.x, threadIdx.x, aggregate, start, mid, end);

        __syncthreads();
        // if(aggregate > 64)
        //     printf("past syncthreads start: %d end: %d\n", start, end);
        //recursively merge up
        if(start != 0 || end != size) {
            // if(threadIdx.x == 0)
            //     printf("agg: %d\n", aggregate*2);
            mergeUp(data, workArr, size, aggregate*2);
        }
        // else {
        //     printf("exiting");
        // }
    }
}

__device__ void recursive_merge(double *data, double *workArr, int size, int aggregate) {
    int taskid = threadIdx.x;
    int blockLimit = (size / (blockDim.x * blockDim.x) < 1 ? 1 : size / (blockDim.x * blockDim.x));
    int blocks = blockLimit > 2048 ? 2048 : blockLimit;
    int threadSize = size / (blocks * blockDim.x);
    int blockstart = blockIdx.x * (size / blocks);
    
    // printf("in recursive_merge");
    // if(threadIdx.x == 0) {
        // for(int i = blockstart; i < (blockstart + (blockDim.x * blockDim.x)); i++) {
            // printf("blockid: %d, taskid: %d - start at %d\n", blockIdx.x, taskid, blockstart);
        // }
    // }
    // printf("out if statement - block %d thread %d agg %d\n", blockIdx.x, threadIdx.x, aggregate);
    if(taskid%aggregate == 0) {
        // printf("in if statement - block %d thread %d agg %d\n", blockIdx.x, threadIdx.x, aggregate);
        // sequential merge with start = taskid * (size / blockDim.x), mid = (taskid + 1) * (size / blockDim.x), end = (taskid + (aggregate/2)) * (size / blockDim.x)
        int start = blockstart + (taskid * threadSize);
        int end = blockstart + ((taskid + aggregate) * threadSize);
        int mid = ((end - start)/2) + start;
        // if(threadIdx.x == 8 && blockIdx.x == 4) 
        //     printf("blockid: %d, taskid: %d, start: %d, mid: %d, end: %d\n", blockIdx.x, threadIdx.x, start, mid, end);
        // for(int i = blockstart; i < (blockstart + (blockDim.x * blockDim.x)); i++) {
        // printf("RecursiveMerge: Block %d Thread %d Agg %d - start: %d mid: %d end: %d\n", blockIdx.x, threadIdx.x, aggregate, start, mid, end);
        // }
        // if(end > size || start < 0) {
        //     printf("issues with sizes in recursive start: %d end: %d\n", start, end);
        // }
        // if(threadIdx.x == 8 && blockIdx.x == 4) {
        //     for(int i = start; i < end; i++) {
        //         printf("Recursive Merge: blockid: %d, taskid: %d - b at %d: %f\n", blockIdx.x, taskid, i, data[i]);
        //     }
        // }
        mergeTwoSortedArrays(data, workArr, start, mid, end);
        // if(threadIdx.x == 8 && blockIdx.x == 4) {
        //     for(int i = start; i < end; i++) {
        //         printf("blockid: %d, taskid: %d - b at %d: %f\n", blockIdx.x, taskid, i, data[i]);
        //     }
        // }

        __syncthreads();
        
        
        //recursively merge up
        if((taskid + (aggregate)) * (size / blockDim.x) != size) {
            recursive_merge(data, workArr, size, aggregate*2);
        }

        // if(aggregate == 16) {
        //     for(int i = blockstart; i < (blockstart + (blockDim.x * blockDim.x)); i++) {
        //         printf("blockid: %d, taskid: %d - b at %d: %f\n", blockIdx.x, taskid, i, data[i]);
        //     }
        // }
    }
}

// Define the CUDA kernel for performing merge sort on each block
__global__ void mergeSortKernel(double *data, double *workArr, int size) {
    // printf("in merge sort kernel\n");
    //figure out which section of the shared, block-level data we're working with
    int taskid = threadIdx.x;
    int blockLimit = (size / (blockDim.x * blockDim.x) < 1 ? 1 : size / (blockDim.x * blockDim.x));
    int blocks = blockLimit > 2048 ? 2048 : blockLimit;
    int threadSize = size / (blocks * blockDim.x);
    int blockstart = blockIdx.x * (size / blocks);
    int start = blockstart + (taskid * threadSize);
    int end = blockstart + ((taskid+1) * threadSize);
    // printf("blockid: %d, taskid: %d, start: %d, end: %d\n", blockIdx.x, threadIdx.x, start, end);
    // if(end > size || start < 0) {
    //     printf("issues with sizes start: %d end: %d\n", start, end);
    // }
    // int start = 0;
    // int end = 32;
    // printf("blockid: %d, taskid: %d, start: %d, end: %d\n", blockIdx.x, taskid, start, end);
    // for(int i = start; i < end; i++) {
        // printf("blockid: %d, taskid: %d - b at %d: %f\n", blockIdx.x, taskid, i, data[i]);
    // }

    // if(blockIdx.x == 0 && taskid == 0) {
    //     for(int i = 0; i < size; i++) {
    //         printf("blockid: %d, taskid: %d - b at %d: %f\n", blockIdx.x, taskid, i, data[i]);
    //     }
    // }

    // if(threadIdx.x == 0) {
    //     for(int i = start; i < end; i++) {
    //         printf("blockid: %d, taskid: %d - b at %d: %f\n", blockIdx.x, taskid, i, data[i]);
    //     }
    // }

    //sort data in our little subsection
    deathSort(data, start, end);

    __syncthreads();
    // if(threadIdx.x == 0) {
    //     for(int i = start; i < end; i++) {
    //         printf("blockid: %d, taskid: %d - b at %d: %f\n", blockIdx.x, taskid, i, data[i]);
    //     }
    // }
    // if(threadIdx.x == 8 && blockIdx.x == 4) {
    //     printf("finished init sort\n");
    //     for(int i = start; i < end; i++) {
    //         printf("blockid: %d, taskid: %d - b at %d: %f\n", blockIdx.x, taskid, i, data[i]);
    //     }
    // }

    
    //recursive merge up
    recursive_merge(data, workArr, size, 2);
    // if(threadIdx.x == 8 && blockIdx.x == 4) {
    //     for(int i = start; i < end; i++) {
    //         printf("blockid: %d, taskid: %d - b at %d: %f\n", blockIdx.x, taskid, i, data[i]);
    //     }
    // }
}

// Define the CUDA kernel for merging sorted blocks, should be performed on a single block theoretically with threads = num_blocks
__global__ void mergeKernel(double *data, double *workArr, int size) {
    // if(blockIdx.x == 0 && threadIdx.x == 0) {
    //     printf("in merge kernel\n");
    // }
    //recursive merge up
    // if(blockIdx.x == 0 && threadIdx.x == 0) {
        // for(int i = 0; i < size; i++) {
        //     printf("blockid: %d, taskid: %d - b at %d: %f\n", blockIdx.x, threadIdx.x, i, data[i]);
        // }
    // }
    mergeUp(data, workArr, size, 1);
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
    BLOCKS = std::min(std::max(sizeOfMatrix / (NUMTHREADS * NUMTHREADS),1),2048);
    gpuErrchk( cudaDeviceSetLimit(cudaLimitStackSize, 4096));

    double *localArr = (double*)malloc(sizeOfMatrix * sizeof(double));
    double* remoteArr;
    gpuErrchk( cudaMalloc((void**)&remoteArr, sizeOfMatrix*sizeof(double)));
    double *workArr;
    gpuErrchk( cudaMalloc((void**)&workArr, sizeOfMatrix*sizeof(double)));

    CALI_MARK_BEGIN(data_init);
    fill_data(localArr, data_order, sizeOfMatrix);
    CALI_MARK_END(data_init);

    // for(int i = 0; i < sizeOfMatrix; i++) {
    //     printf("b at %d: %f\n", i, localArr[i]);
    // }

    CALI_MARK_BEGIN(comm);
    CALI_MARK_BEGIN(comm_large);
    gpuErrchk( cudaMemcpy(remoteArr, localArr, sizeOfMatrix*sizeof(double), cudaMemcpyHostToDevice));
    CALI_MARK_END(comm_large);
    CALI_MARK_END(comm);

    CALI_MARK_BEGIN(comm);
    CALI_MARK_BEGIN(comm_large);
    gpuErrchk( cudaMemcpy(workArr, localArr, sizeOfMatrix*sizeof(double), cudaMemcpyHostToDevice));
    CALI_MARK_END(comm_large);
    CALI_MARK_END(comm);

    dim3 blocks(BLOCKS,1);    /* Number of blocks   */
    dim3 threads(std::min(NUMTHREADS,512),1);  /* Number of threads  */
    dim3 kernelBlock(1,1);
    dim3 kernelThreads(std::min(std::max(BLOCKS/2,1),1024),1);

    printf("blocks: %d threads: %d kernelthreads: %d\n", BLOCKS, std::min(NUMTHREADS,512), std::min(std::max(BLOCKS/2,1),1024));

    // Launch the CUDA kernel for performing merge sort on each block
    CALI_MARK_BEGIN(comp);
    CALI_MARK_BEGIN(comp_large);
    mergeSortKernel<<<BLOCKS,threads>>>(remoteArr,workArr,sizeOfMatrix);
    CALI_MARK_END(comp_large);
    CALI_MARK_END(comp);

    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());

    // Launch the CUDA kernel for merging the sorted blocks
    CALI_MARK_BEGIN(comp);
    CALI_MARK_BEGIN(comp_large);
    mergeKernel<<<kernelBlock,kernelThreads>>>(remoteArr,workArr,sizeOfMatrix);
    CALI_MARK_END(comp_large);
    CALI_MARK_END(comp);

    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());

    // Copy the sorted data back from the GPU to the host
    CALI_MARK_BEGIN(comm);
    CALI_MARK_BEGIN(comm_large);
    gpuErrchk( cudaMemcpy(localArr, remoteArr, sizeOfMatrix*sizeof(double), cudaMemcpyDeviceToHost));
    CALI_MARK_END(comm_large);
    CALI_MARK_END(comm);
    
    // Free memory on the GPU
    cudaFree(remoteArr);
    cudaFree(workArr);

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
            // for(int i = 0; i < sizeOfMatrix; i++) {
            //     printf("b at %d: %f\n", i, localArr[i]);
            // }
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
