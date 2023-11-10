#include "mpi.h"
#include <stdlib.h>
#include <stdio.h>
#include <limits.h>
#include <mpi.h>
#include <time.h>
#include <adiak.hpp>
#include <caliper/cali.h>
#include <caliper/cali-manager.h>

#define MASTER 0               /* taskid of first task */
#define FROM_MASTER 1          /* setting a message type */
#define FROM_WORKER 2          /* setting a message type */

int main (int argc, char *argv[])
{
CALI_CXX_MARK_FUNCTION;
    
int sizeOfArray;
if (argc == 2)
{
    sizeOfArray = atoi(argv[1]);
}
else
{
    printf("\n Please provide the size of the array");
    return 0;
}

int	numtasks,              /* number of tasks in partition */
	taskid,                /* a task identifier */
	numworkers,            /* number of worker tasks */
	source,                /* task id of message source */
	dest,                  /* task id of message destination */
	mtype,                 /* message type */
	rows,                  /* rows of array A sent to each worker */
	averow, extra, offset, /* used to determine rows sent to each worker */
	i, j, k, rc;           /* misc */

double a[sizeOfArray],     /* array 1 */
	b[sizeOfArray];        /* array 2 */                                                       // get rid of b?

MPI_Status status;

/* Define Caliper region names */
const char* whole_computation = "whole_computation";
const char* data_init = "data_init";
const char* comm = "comm";
const char* comm_small = "comm_small";
const char* comm_large = "comm_large";
const char* comp = "comp";
const char* comp_small = "comp_small";
const char* comp_large = "comp_large";
const char* correctness_check = "correctness_check";

MPI_Init(&argc,&argv);
MPI_Comm_rank(MPI_COMM_WORLD,&taskid);
MPI_Comm_size(MPI_COMM_WORLD,&numtasks);

if (numtasks < 2 ) {
    printf("Need at least two MPI tasks. Quitting...\n");
    MPI_Abort(MPI_COMM_WORLD, rc);
    exit(1);
}
numworkers = numtasks-1;

MPI_Comm new_comm;
// MPI_Comm_split(MPI_COMM_WORLD, (taskid != MASTER), taskid, &new_comm);                             TRY IF THINGS GO WRONG
MPI_Comm_split(MPI_COMM_WORLD, (taskid == MASTER) ? MPI_UNDEFINED : 0, 0, &new_comm);

// WHOLE PROGRAM COMPUTATION PART STARTS HERE
CALI_MARK_BEGIN(whole_computation);

// Create caliper ConfigManager object
cali::ConfigManager mgr;
mgr.start();

/**************************** master task ************************************/
if (taskid == MASTER)
{
    printf("sample sort has started with %d tasks.\n",numtasks);
    printf("Initializing arrays...\n");

    CALI_MARK_BEGIN(data_init); // Don't time printf                                            MOVE

    // for(int i = 0; i < sizeOfArray; i++) {
    //     b[i] = srand(1);                                                                 GET RID OF B?
    // }
    
    CALI_MARK_END(data_init);

    /* Send matrix data to the worker tasks */
    averow = sizeOfArray/numworkers;
    extra = sizeOfArray%numworkers;
    offset = 0;
    mtype = FROM_MASTER;

    CALI_MARK_BEGIN(comm);
    CALI_MARK_BEGIN(comm_large);

    for (dest=1; dest<=numworkers; dest++)
    {
        rows = (dest <= extra) ? averow+1 : averow;   	
        printf("Sending %d rows to task %d offset=%d\n",rows,dest,offset);
        MPI_Send(&offset, 1, MPI_INT, dest, mtype, MPI_COMM_WORLD);
        MPI_Send(&rows, 1, MPI_INT, dest, mtype, MPI_COMM_WORLD);
        MPI_Send(&a[offset], rows*sizeOfArray, MPI_DOUBLE, dest, mtype, MPI_COMM_WORLD);
        //MPI_Send(&b, sizeOfArray*sizeOfArray, MPI_DOUBLE, dest, mtype, MPI_COMM_WORLD);
        offset = offset + rows;
    }

    CALI_MARK_END(comm_large);
    CALI_MARK_END(comm);
}


/**************************** worker task ************************************/
if (taskid > MASTER)
{
    mtype = FROM_MASTER;

    CALI_MARK_BEGIN(comm);
    CALI_MARK_BEGIN(comm_small);

    MPI_Recv(&offset, 1, MPI_INT, MASTER, mtype, MPI_COMM_WORLD, &status);
    MPI_Recv(&rows, 1, MPI_INT, MASTER, mtype, MPI_COMM_WORLD, &status);
    MPI_Recv(&a, rows, MPI_DOUBLE, MASTER, mtype, MPI_COMM_WORLD, &status);
    //MPI_Recv(&b, sizeOfArray*sizeOfArray, MPI_DOUBLE, MASTER, mtype, MPI_COMM_WORLD, &status);

    CALI_MARK_END(comm_small);
    CALI_MARK_END(comm);

    // CALI_MARK_BEGIN(comp);
    // CALI_MARK_BEGIN(comp_small);

    // smallSort(a, 0, sizeof(a)/sizeof(a[0]) - 1); //                                                replace with numtasks-2?

    // CALI_MARK_END(comp_small);
    // CALI_MARK_END(comp);

    CALI_MARK_BEGIN(comp);
    CALI_MARK_BEGIN(comp_large);

    sampleSort(a, sizeOfArray, sizeOfArray/(2*numtasks), numtasks); // function to implement                         use array b?

    CALI_MARK_END(comp_large);
    CALI_MARK_END(comp);
}

bool sorted = true;

CALI_MARK_BEGIN(correctness_check);

for (int i = 1; i < sizeOfArray; i++) {
    if (a[i] < a[i-1]) {
        printf("Error. Out of order sequence: %d found\n", a[i]);
        sorted = false;
    }
}
if (sorted) {
    printf("Array is in sorted order\n");
}

CALI_MARK_END(correctness_check);
CALI_MARK_END(whole_computation);

adiak::init(NULL);
adiak::launchdate();    // launch date of the job
adiak::libraries();     // Libraries used
adiak::cmdline();       // Command line used to launch the job
adiak::clustername();   // Name of the cluster
adiak::value("Algorithm", "SampleSort"); // The name of the algorithm you are using (e.g., "MergeSort", "BitonicSort")
adiak::value("ProgrammingModel", "MPI"); // e.g., "MPI", "CUDA", "MPIwithCUDA"
adiak::value("Datatype", "Double"); // The datatype of input elements (e.g., double, int, float)
adiak::value("SizeOfDatatype", sizeof(double)); // sizeof(datatype) of input elements in bytes (e.g., 1, 2, 4)
adiak::value("InputSize", sizeOfArray); // The number of elements in input dataset (1000)
adiak::value("InputType", "Random"); // For sorting, this would be "Sorted", "ReverseSorted", "Random", "1%perturbed"
adiak::value("num_procs", numtasks); // The number of processors (MPI ranks)
//adiak::value("num_threads", num_threads); // The number of CUDA or OpenMP threads
//adiak::value("num_blocks", num_blocks); // The number of CUDA blocks 
adiak::value("group_num", 25); // The number of your group (integer, e.g., 1, 10)
adiak::value("implementation_source", "AI/Handwritten") // Where you got the source code of your algorithm; choices: ("Online", "AI", "Handwritten").

MPI_Comm_free(&new_comm);

// if (taskid == 0)
// {
//     // Master Times
//     printf("******************************************************\n");
//     printf("Master Times:\n");
//     printf("Whole Computation Time: %f \n", whole_computation_time);
//     printf("Master Initialization Time: %f \n", master_initialization_time);
//     printf("Master Send and Receive Time: %f \n", master_send_receive_time);
//     printf("\n******************************************************\n");

//     // Add values to Adiak
//     adiak::value("MPI_Reduce-whole_computation_time", whole_computation_time);
//     adiak::value("MPI_Reduce-master_initialization_time", master_initialization_time);
//     adiak::value("MPI_Reduce-master_send_receive_time", master_send_receive_time);

//     // Must move values to master for adiak
//     mtype = FROM_WORKER;
//     MPI_Recv(&worker_receive_time_max, 1, MPI_DOUBLE, 1, mtype, MPI_COMM_WORLD, &status);
//     MPI_Recv(&worker_receive_time_min, 1, MPI_DOUBLE, 1, mtype, MPI_COMM_WORLD, &status);
//     MPI_Recv(&worker_receive_time_average, 1, MPI_DOUBLE, 1, mtype, MPI_COMM_WORLD, &status);
//     MPI_Recv(&worker_calculation_time_max, 1, MPI_DOUBLE, 1, mtype, MPI_COMM_WORLD, &status);
//     MPI_Recv(&worker_calculation_time_min, 1, MPI_DOUBLE, 1, mtype, MPI_COMM_WORLD, &status);
//     MPI_Recv(&worker_calculation_time_average, 1, MPI_DOUBLE, 1, mtype, MPI_COMM_WORLD, &status);
//     MPI_Recv(&worker_send_time_max, 1, MPI_DOUBLE, 1, mtype, MPI_COMM_WORLD, &status);
//     MPI_Recv(&worker_send_time_min, 1, MPI_DOUBLE, 1, mtype, MPI_COMM_WORLD, &status);
//     MPI_Recv(&worker_send_time_average, 1, MPI_DOUBLE, 1, mtype, MPI_COMM_WORLD, &status);

//     adiak::value("MPI_Reduce-worker_receive_time_max", worker_receive_time_max);
//     adiak::value("MPI_Reduce-worker_receive_time_min", worker_receive_time_min);
//     adiak::value("MPI_Reduce-worker_receive_time_average", worker_receive_time_average);
//     adiak::value("MPI_Reduce-worker_calculation_time_max", worker_calculation_time_max);
//     adiak::value("MPI_Reduce-worker_calculation_time_min", worker_calculation_time_min);
//     adiak::value("MPI_Reduce-worker_calculation_time_average", worker_calculation_time_average);
//     adiak::value("MPI_Reduce-worker_send_time_max", worker_send_time_max);
//     adiak::value("MPI_Reduce-worker_send_time_min", worker_send_time_min);
//     adiak::value("MPI_Reduce-worker_send_time_average", worker_send_time_average);
// }
// else if (taskid == 1)
// { // Print only from the first worker.
//     // Print out worker time results.
      
//     // Compute averages after MPI_Reduce
//     worker_receive_time_average = worker_receive_time_sum / (double)numworkers;
//     worker_calculation_time_average = worker_calculation_time_sum / (double)numworkers;
//     worker_send_time_average = worker_send_time_sum / (double)numworkers;

//     printf("******************************************************\n");
//     printf("Worker Times:\n");
//     printf("Worker Receive Time Max: %f \n", worker_receive_time_max);
//     printf("Worker Receive Time Min: %f \n", worker_receive_time_min);
//     printf("Worker Receive Time Average: %f \n", worker_receive_time_average);
//     printf("Worker Calculation Time Max: %f \n", worker_calculation_time_max);
//     printf("Worker Calculation Time Min: %f \n", worker_calculation_time_min);
//     printf("Worker Calculation Time Average: %f \n", worker_calculation_time_average);
//     printf("Worker Send Time Max: %f \n", worker_send_time_max);
//     printf("Worker Send Time Min: %f \n", worker_send_time_min);
//     printf("Worker Send Time Average: %f \n", worker_send_time_average);
//     printf("\n******************************************************\n");

//     mtype = FROM_WORKER;
//     MPI_Send(&worker_receive_time_max, 1, MPI_DOUBLE, MASTER, mtype, MPI_COMM_WORLD);
//     MPI_Send(&worker_receive_time_min, 1, MPI_DOUBLE, MASTER, mtype, MPI_COMM_WORLD);
//     MPI_Send(&worker_receive_time_average, 1, MPI_DOUBLE, MASTER, mtype, MPI_COMM_WORLD);
//     MPI_Send(&worker_calculation_time_max, 1, MPI_DOUBLE, MASTER, mtype, MPI_COMM_WORLD);
//     MPI_Send(&worker_calculation_time_min, 1, MPI_DOUBLE, MASTER, mtype, MPI_COMM_WORLD);
//     MPI_Send(&worker_calculation_time_average, 1, MPI_DOUBLE, MASTER, mtype, MPI_COMM_WORLD);
//     MPI_Send(&worker_send_time_max, 1, MPI_DOUBLE, MASTER, mtype, MPI_COMM_WORLD);
//     MPI_Send(&worker_send_time_min, 1, MPI_DOUBLE, MASTER, mtype, MPI_COMM_WORLD);
//     MPI_Send(&worker_send_time_average, 1, MPI_DOUBLE, MASTER, mtype, MPI_COMM_WORLD);
// }

// Flush Caliper output before finalizing MPI
mgr.stop();
mgr.flush();

MPI_Finalize();
};

void smallSort(double arr[], int left, int right) {
    if (left < right) {
        // Partition the array
        double pivot = arr[right];
        int i = left - 1;
        for (int j = left; j <= right - 1; j++) {
            if (arr[j] < pivot) {
                i++;
                double temp = arr[i];
                arr[i] = arr[j];
                arr[j] = temp;
            }
        }
        double temp = arr[i + 1];
        arr[i + 1] = arr[right];
        arr[right] = temp;
        int pi = i + 1;
        // Recursively sort elements before and after partition
        smallSort(arr, left, pi - 1);
        smallSort(arr, pi + 1, right);
    }
};

void sampleSort(double arr[], int n, int k, int p) {
    CALI_CXX_MARK_FUNCTION;
    const int threshold = 10; // Define the threshold for smallSort

    // Step 1
    if (n / k < threshold) {
        CALI_MARK_BEGIN(comp_small);
        smallSort(arr, 0, n - 1); // Use smallSort if average bucket size is below threshold
        CALI_MARK_END(comp_small);
        return;
    }

    // Select samples and sort them
    double S[p*k];
    for (int i = 0; i < p*k; i++) {
        S[i] = arr[rand() % n]; // Randomly select samples from A
    }
    smallSort(S, 0, p*k - 1); // Sort the samples

    // Define splitters
    double splitters[p+1];
    splitters[0] = -INFINITY;
    for (int i = 1; i < p; i++) {
        splitters[i] = S[i*k];
    }
    splitters[p] = INFINITY;

    // Step 2
    std::vector<std::vector<double>> buckets(p);

    for (int i = 0; i < n; i++) {
        for (int j = 0; j < p; j++) {
            if (arr[i] > splitters[j] && arr[i] <= splitters[j+1]) {
                buckets[j].push_back(arr[i]);
                break;
            }
        }
    }

    // Step 3 and concatenation
    for (int i = 0; i < p; i++) {
        sampleSort(buckets[i].data(), buckets[i].size(), k, p);
        if (i != p-1) {
            std::copy(buckets[i].begin(), buckets[i].end(), arr + i*k);
        } else {
            std::copy(buckets[i].begin(), buckets[i].end(), arr + i*k);
        }
    }
};