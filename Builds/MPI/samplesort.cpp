#include "mpi.h"
#include <stdlib.h>
#include <stdio.h>
#include <limits.h>
#include <mpi.h>
#include <time.h>
#include <adiak.hpp>
#include <caliper/cali.h>
#include <caliper/cali-manager.h>

// Add other necessary headers if required

#define threshold 10 // Adjust this threshold as needed

// Function to initialize data
void dataInit(int* A, int arr_size) {
    CALI_MARK_BEGIN("data_init");
    srand(time(NULL));
    int i;
    for (i = 0; i < arr_size; ++i) {
        A[i] = rand();
    }
    CALI_MARK_END("data_init");
}

// Function for small sorting using quicksort
void smallSort(int A[], int left, int right) {
    CALI_MARK_BEGIN("comp");
    CALI_MARK_BEGIN("comp_small");

    if (left < right) {
        // Partition the array
        int pivot = A[right];
        int i = left - 1;
        for (int j = left; j <= right - 1; j++) {
            if (A[j] < pivot) {
                i++;
                int temp = A[i];
                A[i] = A[j];
                A[j] = temp;
            }
        }
        int temp = A[i + 1];
        A[i + 1] = A[right];
        A[right] = temp;
        int pi = i + 1;
        // Recursively sort elements before and after partition
        smallSort(A, left, pi - 1);
        smallSort(A, pi + 1, right);
    }

    CALI_MARK_END("comp_small");
    CALI_MARK_END("comp");
}

// Function for sampleSort
void sampleSort(int A[], int arr_size, int k, int num_proc, int rank, int size) {
    // if average bucket size is below a threshold switch to quicksort
    if (arr_size / k < threshold) {
        smallSort(A, 0, arr_size - 1); // Sort the entire array
        return;
    }
printf("3\n");
    // Step 1
    CALI_MARK_BEGIN("comp");
    CALI_MARK_BEGIN("comp_large");

    int* S = (int*)malloc(num_proc * k * sizeof(int));

    // Select and sort samples
    for (int i = 1; i < num_proc; i++) {
        int sample_index = i * (arr_size / num_proc); // Calculate the sample index
        S[i - 1] = A[sample_index]; // Select the sample
    }
    smallSort(S, 0, num_proc - 2); // Sort the samples using smallSort
printf("4\n");
    // Select splitters
    int* splitters = (int*)malloc(num_proc * sizeof(int));
    splitters[0] = -INT_MAX;
    splitters[num_proc - 1] = INT_MAX;
    for (int i = 1; i < num_proc; i++) {
        int splitter_index = i * (num_proc - 1); // Calculate the splitter index
        splitters[i] = S[splitter_index]; // Select the splitter
    }

    CALI_MARK_END("comp_large");

   // Step 2
    int* bucket_sizes = (int*)malloc(num_proc * sizeof(int)); // Array to store the size of each bucket
printf("5\n");
    // Initialize bucket sizes to zero
    for (int i = 0; i < num_proc; i++) {
        bucket_sizes[i] = 0;
    }

    // Count how many elements belong to each bucket
    for (int i = 0; i < arr_size; i++) {
        int j;
        for (j = 0; j < num_proc; j++) {
            if (splitters[j] < A[i] && A[i] <= splitters[j + 1]) {
                bucket_sizes[j]++;
                break;
            }
        }
    }
printf("6\n");
    // Calculate the displacement for each bucket
    int* bucket_displacements = (int*)malloc(num_proc * sizeof(int));
    bucket_displacements[0] = 0;
    for (int i = 1; i < num_proc; i++) {
        bucket_displacements[i] = bucket_displacements[i - 1] + bucket_sizes[i - 1];
    }

    // Allocate memory for the local bucket
    int local_bucket_size = bucket_sizes[rank];
    int* local_bucket = (int*)malloc(local_bucket_size * sizeof(int));

    // Place elements into the local bucket based on the splitters
    int local_index = 0;
    for (int i = 0; i < arr_size; i++) {
        int j;
        for (j = 0; j < num_proc; j++) {
            if (splitters[j] < A[i] && A[i] <= splitters[j + 1]) {
                local_bucket[local_index] = A[i];
                local_index++;
                break;
            }
        }
    }
printf("7\n");
    CALI_MARK_END("comp");

    // Step 3 and concatenation
    // Determine the size of each local bucket
    local_bucket_size = bucket_sizes[rank];
    
    // Create an array to hold the sizes of all local buckets
    int* all_bucket_sizes = (int*)malloc(num_proc * sizeof(int));
    
    // Use MPI_Allgather to share the local bucket sizes with all processes
    MPI_Allgather(&local_bucket_size, 1, MPI_INT, all_bucket_sizes, 1, MPI_INT, MPI_COMM_WORLD);
    
    // Calculate the total size of all buckets
    int total_bucket_size = 0;
    for (int i = 0; i < num_proc; i++) {
        total_bucket_size += all_bucket_sizes[i];
    }
    printf("8\n");
    // Allocate memory for the concatenated array
    int* concatenated_array = (int*)malloc(total_bucket_size * sizeof(int));
    
    // Use MPI_Alltoallv to exchange data between processes
    MPI_Alltoallv(local_bucket, bucket_sizes, bucket_displacements, MPI_INT,
                concatenated_array, all_bucket_sizes, bucket_displacements, MPI_INT, MPI_COMM_WORLD);
    
    // Now 'concatenated_array' contains the sorted elements from all buckets
    // You can use it as needed
    
    // Don't forget to free allocated memory when done
    free(S);
    free(splitters);
    free(bucket_sizes);
    free(bucket_displacements);
    free(local_bucket);
    free(all_bucket_sizes);
    free(concatenated_array);
}

int main(int argc, char** argv) {
    CALI_CXX_MARK_FUNCTION;
printf("0\n");
    // const char* comm = "comm";
    // const char* comm_large = "comm_large";
    // const char* comm_small = "comm_small";
    // const char* correctness_check = "correctness_check";
    // const char* mpi_barrier = "mpi_barrier";

    CALI_MARK_BEGIN("whole_computation");
    MPI_Init(&argc, &argv);

    printf("1\n");

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (argc != 3) {
        if (rank == 0) {
            printf("Usage: %s <array_size> <num_processes>\n", argv[0]);
        }
        MPI_Finalize();
        return 1;
    }

    int arr_size = atoi(argv[1]);
    int num_proc = atoi(argv[2]);
    int k = arr_size / (2 * num_proc); // change?

    int* A = (int*)malloc(arr_size * sizeof(int));

    dataInit(A, arr_size);

    printf("2\n");

    sampleSort(A, arr_size, k, num_proc, rank, size);

    adiak::init(NULL);
    adiak::launchdate();    // launch date of the job
    adiak::libraries();     // Libraries used
    adiak::cmdline();       // Command line used to launch the job
    adiak::clustername();   // Name of the cluster
    adiak::value("Algorithm", "SampleSort"); // The name of the algorithm you are using (e.g., "MergeSort", "BitonicSort")
    adiak::value("ProgrammingModel", "MPI"); // e.g., "MPI", "CUDA", "MPIwithCUDA"
    adiak::value("Datatype", "int"); // The datatype of input elements (e.g., double, int, float)
    adiak::value("SizeOfDatatype", "4"); // sizeof(datatype) of input elements in bytes (e.g., 1, 2, 4)
    adiak::value("InputSize", "1000"); // The number of elements in input dataset (1000)
    adiak::value("InputType", "Random"); // For sorting, this would be "Sorted", "ReverseSorted", "Random", "1%perturbed"
    adiak::value("num_procs", size); // The number of processors (MPI ranks)
    //adiak::value("num_threads", num_threads); // The number of CUDA or OpenMP threads
    //adiak::value("num_blocks", num_blocks); // The number of CUDA blocks 
    adiak::value("group_num", 1); // The number of your group (integer, e.g., 1, 10)
    adiak::value("implementation_source", "AI/Handwritten") // Where you got the source code of your algorithm; choices: ("Online", "AI", "Handwritten").

    free(A);

    MPI_Finalize();
    CALI_MARK_END("whole_computation");
printf("9\n");
    return 0;
}
