#include "mpi.h"
#include <stdio.h>
#include <stdlib.h>
#include <limits.h>
#include <math.h>

#include <caliper/cali.h>
#include <caliper/cali-manager.h>
#include <adiak.hpp>

#define MASTER 0               /* taskid of first task */
#define FROM_MASTER 1          /* setting a message type */
#define FROM_WORKER 2          /* setting a message type */

int main (int argc, char *argv[])
{
CALI_CXX_MARK_FUNCTION;

int sizeOfMatrix;
if (argc == 2)
{
    sizeOfMatrix = atoi(argv[1]);
}
else
{
    printf("\n Please provide the size of the matrix");
    return 0;
}

int	numtasks,              /* number of tasks in partition */
	taskid,                /* a task identifier */
	numworkers,            /* number of worker tasks */
	source,                /* task id of message source */
	dest,                  /* task id of message destination */
	mtype,                 /* message type */
	rows,                  /* rows of matrix A sent to each worker */
	averow, extra, offset, /* used to determine rows sent to each worker */
    double	a[sizeOfMatrix],   /* receiving array*/
	    b[sizeOfMatrix];       /* whole array */
	i, j, k, rc;           /* misc */
    
/* Define Caliper region names */
const char* whole_computation = "whole_computation";
const char* data_init = "data_init";
const char* correctness_check = "correctness_check";
const char* comm = "comm";
const char* comm_small = "comm_small";
const char* comm_large = "comm_large";
const char* comp = "comp";
const char* comp_small = "comp_small";
const char* comp_large = "comp_large";

MPI_Status status;

MPI_Init(&argc,&argv);
MPI_Comm_rank(MPI_COMM_WORLD,&taskid);
MPI_Comm_size(MPI_COMM_WORLD,&numtasks);
if (numtasks < 2 ) {
  printf("Need at least two MPI tasks. Quitting...\n");
  MPI_Abort(MPI_COMM_WORLD, rc);
  exit(1);
  }
numworkers = numtasks-1;


MPI_Comm comm;
MPI_Comm_split(MPI_COMM_WORLD, (taskid == MASTER) ? MPI_UNDEFINED : 0, 0, &comm);

CALI_MARK_BEGIN(whole_computation);

cali::ConfigManager mgr;
mgr.start();
 
    
    if (taskid == MASTER)
    {

        // INITIALIZATION PART FOR THE MASTER PROCESS STARTS HERE

        printf("merge sort has started with %d tasks.\n",numtasks);
        printf("Initializing arrays...\n");
        CALI_MARK_BEGIN(data_init);
        for(int i = 0; i < sizeOfMatrix; i++) {
            b[i] = srand(1);
        }
        CALI_MARK_END(data_init);

        averow = sizeOfMatrix/numworkers;
        extra = sizeOfMatrix%numworkers;
        offset = 0;
        mtype = FROM_MASTER;
        CALI_MARK_BEGIN(comm);
        CALI_MARK_BEGIN(comm_large);
        for(int i = 1; i <= numworkers; i++) {
            rows = (i <= extra) ? averow+1 : averow;   	
            printf("Sending %d rows to task %d offset=%d\n",rows,i,offset);
            MPI_Send(&offset, 1, MPI_INT, i, mtype, MPI_COMM_WORLD);
            MPI_Send(&rows, 1, MPI_INT, i, mtype, MPI_COMM_WORLD);
            MPI_Send(&a[offset], sizeOfMatrix, MPI_DOUBLE, i, mtype,
                    MPI_COMM_WORLD);
            offset += rows;
        }
        CALI_MARK_END(comm_large);
        CALI_MARK_END(comm);
    }

    if(taskid > MASTER) {
        //receive info from master
        mtype = FROM_MASTER;
        CALI_MARK_BEGIN(comm);
        CALI_MARK_BEGIN(comm_small);
        MPI_Recv(&offset, 1, MPI_INT, MASTER, mtype, MPI_COMM_WORLD, &status);
        MPI_Recv(&rows, 1, MPI_INT, MASTER, mtype, MPI_COMM_WORLD, &status);
        MPI_Recv(&a, rows, MPI_DOUBLE, MASTER, mtype, MPI_COMM_WORLD, &status);
        CALI_MARK_END(comm_small);
        CALI_MARK_END(comm);
        //sort dataset
        CALI_MARK_BEGIN(comp);
        CALI_MARK_BEGIN(comp_small);
        qsort((void*)a, rows, sizeof(double), compInt);
        CALI_MARK_END(comp);
        CALI_MARK_END(comp_small);
        //merge up
        recursive_merge(a,b,offset,rows,2);
    }

    //check array order
    assert(test_array_is_in_order(b));

    //END COMPUTATION
    CALI_MARK_END(whole_computation);

    adiak::init(NULL);
    adiak::launchdate();    // launch date of the job
    adiak::libraries();     // Libraries used
    adiak::cmdline();       // Command line used to launch the job
    adiak::clustername();   // Name of the cluster
    adiak::value("Algorithm", "MergeSort"); // The name of the algorithm you are using (e.g., "MergeSort", "BitonicSort")
    adiak::value("ProgrammingModel", "MPI"); // e.g., "MPI", "CUDA", "MPIwithCUDA"
    adiak::value("Datatype", "Double"); // The datatype of input elements (e.g., double, int, float)
    adiak::value("SizeOfDatatype", sizeof(Double)); // sizeof(datatype) of input elements in bytes (e.g., 1, 2, 4)
    adiak::value("InputSize", sizeOfMatrix); // The number of elements in input dataset (1000)
    adiak::value("InputType", inputType); // For sorting, this would be "Sorted", "ReverseSorted", "Random", "1%perturbed"
    adiak::value("num_procs", num_tasks); // The number of processors (MPI ranks)
    // adiak::value("num_threads", num_threads); // The number of CUDA or OpenMP threads
    // adiak::value("num_blocks", num_blocks); // The number of CUDA blocks 
    adiak::value("group_num", group_number); // The number of your group (integer, e.g., 1, 10)
    adiak::value("implementation_source", "Handwritten") // Where you got the source code of your algorithm; choices: ("Online", "AI", "Handwritten").

    double worker_receive_time_max,
      worker_receive_time_min,
      worker_receive_time_sum,
      worker_receive_time_average,
      worker_calculation_time_max,
      worker_calculation_time_min,
      worker_calculation_time_sum,
      worker_calculation_time_average,
      worker_send_time_max,
      worker_send_time_min,
      worker_send_time_sum,
      worker_send_time_average = 0; 

    if (taskid == 0)
   {
      // Master Times
      printf("******************************************************\n");
      printf("Master Times:\n");
      printf("Whole Computation Time: %f \n", whole_computation_time);
      printf("Master Initialization Time: %f \n", master_initialization_time);
      printf("Master Send and Receive Time: %f \n", master_send_receive_time);
      printf("\n******************************************************\n");

      // Add values to Adiak
      adiak::value("MPI_Reduce-whole_computation_time", whole_computation_time);
      adiak::value("MPI_Reduce-master_initialization_time", master_initialization_time);
      adiak::value("MPI_Reduce-master_send_receive_time", master_send_receive_time);

      // Must move values to master for adiak
      mtype = FROM_WORKER;
      MPI_Recv(&worker_receive_time_max, 1, MPI_DOUBLE, 1, mtype, MPI_COMM_WORLD, &status);
      MPI_Recv(&worker_receive_time_min, 1, MPI_DOUBLE, 1, mtype, MPI_COMM_WORLD, &status);
      MPI_Recv(&worker_receive_time_average, 1, MPI_DOUBLE, 1, mtype, MPI_COMM_WORLD, &status);

      adiak::value("MPI_Reduce-worker_receive_time_max", worker_receive_time_max);
      adiak::value("MPI_Reduce-worker_receive_time_min", worker_receive_time_min);
      adiak::value("MPI_Reduce-worker_receive_time_average", worker_receive_time_average);
   }
   else if (taskid == 1)
   { // Print only from the first worker.
      // Print out worker time results.
      
      // Compute averages after MPI_Reduce
      worker_receive_time_average = worker_receive_time_sum / (double)numworkers;

      printf("******************************************************\n");
      printf("Worker Times:\n");
      printf("Worker Receive Time Max: %f \n", worker_receive_time_max);
      printf("Worker Receive Time Min: %f \n", worker_receive_time_min);
      printf("Worker Receive Time Average: %f \n", worker_receive_time_average);
      printf("\n******************************************************\n");

      mtype = FROM_WORKER;
      MPI_Send(&worker_receive_time_max, 1, MPI_DOUBLE, MASTER, mtype, MPI_COMM_WORLD);
      MPI_Send(&worker_receive_time_min, 1, MPI_DOUBLE, MASTER, mtype, MPI_COMM_WORLD);
      MPI_Send(&worker_receive_time_average, 1, MPI_DOUBLE, MASTER, mtype, MPI_COMM_WORLD);
   }

   // Flush Caliper output before finalizing MPI
   mgr.stop();
   mgr.flush();

   MPI_Finalize();
};

int compInt(const void* a, const void* b) {
    int arg1 = *(const int*)a;
    int arg2 = *(const int*)b;

    if(arg1 == arg2) {return 0};
    else if(arg1 > arg2) {return 1};
    else {return -1;}
};
//merge 2 sorted lists a,b into a result array c. for this assignment, the arrays should be sequentially stored, with a coming before b. 
//the current call for this invokes a as both a and c, with a being a copy and c being the reference, so updating c should have no effect on a
void merge(double[] a, double[] b, double[] &c, int a_start, int a_size, int b_start, int b_size) {
    int ai = a_start;
    int bi = b_start;
    int ci = 0;
    while(a_size > ai && b_size > bi) {
        if(a[ai] < b[bi]) {
            c[ci] = a[ai];
            ci++;
            ai++;
        }
        else if(a[ai] > b[bi]) {
            c[ci] = b[bi];
            ci++;
            bi++;
        }
        else {
            c[ci] = a[ai];
            c[ci+1] = b[bi];
            c+=2;
            ai++;
            bi++;
        }
        if(ai >= a_size) {
            for(int i = bi; i < b_size; i++) {
                c[ci] = b[i]
                ci++;
            }
            break;
        }
        else if(bi >= b_size) {
            for(int i = ai; i < a_size; i++) {
                c[ci] = a[i]
                ci++;
            }
            break;
        }
    }
};
//agg should start at 2
void recursive_merge(double[] &a, double[] &b, int offset, int rows, int aggregate) {
CALI_CXX_MARK_FUNCTION;
    mtype = FROM_WORKER;
    //if process is a worker at this current level proceed
    if(taskid%aggregate == 1) {
        //store current info before overwriting
        moffset = offset;
        mrows = rows;
        //receive info
        CALI_MARK_BEGIN(comp);
        CALI_MARK_BEGIN(comp_small);
        MPI_Recv(&offset, 1, MPI_INT, taskid + (aggregate/2), mtype, MPI_COMM_WORLD, &status);
        MPI_Recv(&rows, 1, MPI_INT, taskid + (aggregate/2), mtype, MPI_COMM_WORLD, &status);
        MPI_Recv(&a, rows, MPI_DOUBLE, taskid + (aggregate/2), mtype, MPI_COMM_WORLD, &status);
        CALI_MARK_BEGIN(comp);
        CALI_MARK_BEGIN(comp_small);
        //merge incoming with current (leaves offset as moffset in b with length mrows + rows)
        CALI_MARK_BEGIN(comp);
        CALI_MARK_BEGIN(comp_large);
        merge(b, a, b, moffset, mrows, 0, rows);
        CALI_MARK_END(comp);
        CALI_MARK_END(comp_large);
        //recurse a level up
        if(rows + mrows != sizeOfMatrix) {
            recursive_merge(a,b, moffset, mrows + rows, aggregate*2);
        }
    }
    //only half the processes are working, so the other half must send their data to the worker
    else {
        MPI_Send(&offset, 1, MPI_INT, taskid - (aggregate/2), mtype, MPI_COMM_WORLD);
        MPI_Send(&rows, 1, MPI_INT, taskid - (aggregate/2), mtype, MPI_COMM_WORLD);
        MPI_Send(&b[offset], rows, MPI_DOUBLE, taskid - (aggregate/2), mtype, MPI_COMM_WORLD);
    }
};

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