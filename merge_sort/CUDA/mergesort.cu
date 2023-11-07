#include <stdlib.h>
#include <stdio.h>
#include <time.h>

#include <caliper/cali.h>
#include <caliper/cali-manager.h>
#include <adiak.hpp>

#include <cuda.h>
#include <cuda_runtime.h>

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
        merge(data, start, mid, end);
        
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
    int start = taskid * (size / blockDim.x);
    int end = (taskid + 1) * (size / blockDim.x);

    //sort data in our little subsection
    sequential_sort(data[start], end-start);

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
    // Allocate memory for the input data on the host

    NUMTHREADS = atoi(argv[1]);
    sizeOfMatrix = atoi(argv[2]);
    BLOCKS = sizeOfMatrix / NUMTHREADS;

    double[] localArr = new double[sizeOfMatrix];
    double* remoteArr;
    cudaMalloc((void**)&remoteArr, sizeOfMatrix);

    for(int i = 0; i < sizeOfMatrix; i++) {
        localArr[i] = srand(1);
    }

    cudaMemcpy(remoteArr, localArr, sizeOfMatrix, cudaMemcpyHostToDevice);

    dim3 blocks(BLOCKS-1,1);    /* Number of blocks   */
    dim3 threads(NUMTHREADS,1);  /* Number of threads  */
    dim3 kernelBlock(1,1); 

    // Launch the CUDA kernel for performing merge sort on each block
    mergeSortKernel<<<blocks,threads>>>(remoteArr,sizeOfMatrix);

    // Launch the CUDA kernel for merging the sorted blocks
    mergeKernel<<<kernelBlock,threads>>>(remoteArr,sizeOfMatrix);

    // Copy the sorted data back from the GPU to the host
    cudaMemcpy(localArr, remoteArr, sizeOfMatrix, cudaMemcpyDeviceToHost);

    // Free memory on the GPU
    cudaFree(remoteArr);

    // Print the sorted data
    test_array_is_in_order(localArr);

    return 0;
}
