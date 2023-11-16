#include "mpi.h"
#include <stdio.h>
#include <stdlib.h>
#include <cstring>

#include <caliper/cali.h>
#include <caliper/cali-manager.h>
#include <adiak.hpp>

int OPTION;
int NUM_VALS;

const char* options[4] = {"random", "sorted", "reverse_sorted", "1%perturbed"};

float random_float() {
  return (float)rand()/(float)RAND_MAX;
}

void array_fill(float *array, int length, int offset, int choice) {
    if (choice == 1) {
        srand(offset);
        for (int i = 0; i < length; ++i) {
            array[i] = random_float();
        }
    } 
    else if (choice == 2) {
        for (int i = 0; i < length; ++i) {
            array[i] = (float)offset+i;
        }
    } 
    else if (choice == 3) {
        for (int i = 0; i < length; ++i) {
            array[i] = (float)offset+length-1-i;
        }
    } 
    else if (choice == 4) {
        for (int i = 0; i < length; ++i) {
            array[i] = (float)i;
        }

        int p_count = length / 100;
        srand(offset);
        for (int i = 0; i < p_count; ++i) {
            int index = rand() % length;
            array[index] = random_float();
        }
    }
}

int Compare(const void* a, const void* b) {
    return ( *(int*)a - *(int*)b );
}

void merge_low(float *values, float *A, float *B, int vals_numbers) {
   int a = 0;
   int b = 0;
   int c = 0;

   while (c < vals_numbers) {
      if (values[a] <= A[b]) {
         B[c] = values[a];
         c++; a++;
      } 
      else {
         B[c] = A[b];
         c++; b++;
      }
   }

   memcpy(values, B, vals_numbers*sizeof(float));
}

void merge_high(float *array_values, float *A, float *B, int vals_numbers) {
   int a = vals_numbers - 1;
   int b = vals_numbers - 1;
   int c = vals_numbers - 1;

   while (c >= 0) {
      if (array_values[a] >= A[b]) {
         B[c] = array_values[a];
         c--; a--;
      } 
      else {
         B[c] = A[b];
         c--; b--;
      }
   }

   memcpy(array_values, B, vals_numbers*sizeof(float));
}

void odd_even_iter(float *array_values, float *A, float *B, int vals_numbers, int phase, int temp_even, int temp_odd, int rank, int num_threads, MPI_Comm comm) {
   
   MPI_Status status;

    if (phase % 2 != 0) {
      if (temp_odd >= 0) {
         MPI_Sendrecv(array_values, vals_numbers, MPI_FLOAT, temp_odd, 0, A, vals_numbers, MPI_FLOAT, temp_odd, 0, comm, &status);
         if (rank % 2 != 0){
            merge_low(array_values, A, B, vals_numbers);
         }
         else{
            merge_high(array_values, A, B, vals_numbers);
         }
      }
   }

   else {
      if (temp_even >= 0) {
         MPI_Sendrecv(array_values, vals_numbers, MPI_FLOAT, temp_even, 0, A, vals_numbers, MPI_FLOAT, temp_even, 0, comm, &status);
         if (rank % 2 != 0){
            merge_high(array_values, A, B, vals_numbers);
         }
         else{
            merge_low(array_values, A, B, vals_numbers);
         }
      }
   } 
}

void odd_even_sort(float *array_values, int vals_numbers, int rank, int num_threads, MPI_Comm comm) {
    int temp_even;
    int temp_odd;

    float *A = (float*) malloc(vals_numbers*sizeof(float));
    float *B = (float*) malloc(vals_numbers*sizeof(float));

    if (rank % 2 != 0) {
        temp_even = rank - 1;
        temp_odd = rank + 1;
        if (temp_odd == num_threads) temp_odd = MPI_PROC_NULL;
    } else {
        temp_even = rank + 1;
        if (temp_even == num_threads) temp_even = MPI_PROC_NULL;
        temp_odd = rank-1;
    }

    qsort(array_values, vals_numbers, sizeof(int), Compare);

    for (int phase = 0; phase < num_threads; phase++) {
        odd_even_iter(array_values, A, B, vals_numbers, phase, temp_even, temp_odd, rank, num_threads, comm);
    }
    
    free(A);
    free(B);
}

void assign(float* array_values, float* array, int offset, int rank) {
    for (int i = 0; i < offset; ++i) {
        array_values[i+offset*rank] = array[i];
    }
}

int check(float* array_values, int length) {
    for (int i = 1; i < length; ++i) {
        if (array_values[i] < array_values[i -1]) {
            return 0;
        }
    }
    return 1;
}

int main (int argc, char *argv[]) {
    CALI_CXX_MARK_FUNCTION;

    int	numtasks, taskid;
    
    NUM_VALS = atoi(argv[1]);
    OPTION = atoi(argv[2]);

    MPI_Status status;

    MPI_Init(&argc,&argv);
    MPI_Comm_rank(MPI_COMM_WORLD,&taskid);
    MPI_Comm_size(MPI_COMM_WORLD,&numtasks);

    cali::ConfigManager mgr;
    mgr.start(); 

    int offset = NUM_VALS / numtasks;

    float *values = (float*) malloc(offset * sizeof(float));

    CALI_MARK_BEGIN("data_init");
    array_fill(values, offset, offset*taskid, OPTION);
    CALI_MARK_END("data_init");

    CALI_MARK_BEGIN("comp");
    CALI_MARK_BEGIN("comp_large");
    odd_even_sort(values, offset, taskid, numtasks, MPI_COMM_WORLD);
    CALI_MARK_END("comp_large");
    CALI_MARK_END("comp");

    float* global_list;

    CALI_MARK_BEGIN("comm");
    CALI_MARK_BEGIN("comm_large");
    if (taskid == 0) {
        global_list = (float*) malloc( NUM_VALS * sizeof(float));
        assign(global_list, values, offset, 0);
        float *temp = (float*) malloc(offset * sizeof(float));
        for (int rank = 1; rank < numtasks; rank++) {
            CALI_MARK_BEGIN("MPI_Recv");
            MPI_Recv(temp, offset, MPI_FLOAT, rank, 0, MPI_COMM_WORLD, &status);
            CALI_MARK_END("MPI_Recv");
            assign(global_list, temp, offset, rank);
        }
        free(temp);

    } 
    else {
        CALI_MARK_BEGIN("MPI_Send");
        MPI_Send(values, offset, MPI_FLOAT, 0, 0, MPI_COMM_WORLD);
        CALI_MARK_END("MPI_Send");
    }
    CALI_MARK_END("comm_large");
    CALI_MARK_END("comm");

    if (taskid == 0) {
        CALI_MARK_BEGIN("correctness_check");
        int correctness = check(global_list, NUM_VALS);
        CALI_MARK_END("correctness_check");

        adiak::init(NULL);
        adiak::launchdate();    // launch date of the job
        adiak::libraries();     // Libraries used
        adiak::cmdline();       // Command line used to launch the job
        adiak::clustername();   // Name of the cluster
        adiak::value("Algorithm", "odd_even_sort"); // The name of the algorithm you are using (e.g., "MergeSort", "BitonicSort")
        adiak::value("ProgrammingModel", "MPI"); // e.g., "MPI", "CUDA", "MPIwithCUDA"
        adiak::value("Datatype", "float"); // The datatype of input elements (e.g., double, int, float)
        adiak::value("SizeOfDatatype", sizeof(float)); // sizeof(datatype) of input elements in bytes (e.g., 1, 2, 4)
        adiak::value("InputSize", NUM_VALS); // The number of elements in input dataset (1000)
        adiak::value("InputType", options[OPTION-1]); // For sorting, this would be "Sorted", "ReverseSorted", "Random", "1%perturbed"
        adiak::value("num_procs", numtasks); // The number of processors (MPI ranks)
        adiak::value("group_num", 5); // The number of your group (integer, e.g., 1, 10)
        adiak::value("implementation_source", "Online and Handwritten"); // Where you got the source code of your algorithm; choices: ("Online", "AI", "Handwritten").
        adiak::value("correctness", correctness); // Whether the dataset has been sorted (0, 1)
    }

    // Flush Caliper output before finalizing MPI
    mgr.stop();
    mgr.flush();

    MPI_Finalize();
}
