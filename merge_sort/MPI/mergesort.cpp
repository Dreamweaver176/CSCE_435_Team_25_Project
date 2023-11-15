#include "mpi.h"
#include <stdio.h>
#include <stdlib.h>
#include <limits.h>
#include <math.h>
#include <cassert>

#include <caliper/cali.h>
#include <caliper/cali-manager.h>
#include <adiak.hpp>

#define MASTER 0               /* taskid of first task */
#define FROM_MASTER 1          /* setting a message type */
#define FROM_WORKER 2          /* setting a message type */

int sizeOfMatrix;
char data_order;

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
const char* MPI_Recv_Region = "MPI_Recv";
const char* MPI_Send_Region = "MPI_Send";
const char* receive_merged_data = "receive_merged_data";
const char* merge_region  = "merge";
const char* recursive_merge_region = "recursive_merge";
const char* send_merged_data = "send_merged_data";

MPI_Status status;


int compInt(const void* a, const void* b) {
    double arg1 = *(const double*)a;
    double arg2 = *(const double*)b;

    if(arg1 == arg2) {return 0;}
    else if(arg1 > arg2) {return 1;}
    else {return -1;}
};

void merge(double *arr, double *workArr, int start, int middle, int end) {
    int i = start;
    int j = middle;
    
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
}

//merge 2 sorted lists a,b into a result array c. for this assignment, the arrays should be sequentially stored, with a coming before b. 
//the current call for this invokes a as both a and c, with a being a copy and c being the reference, so updating c should have no effect on a
void merge(double a[], double b[], double (&c)[], int a_start, int a_size, int b_start, int b_size, int offset) {
    int ai = 0;
    int bi = 0;
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
            ci+=2;
            ai++;
            bi++;
        }
        if(ai >= a_size) {
            for(int i = bi; i < b_size; i++) {
                c[ci] = b[i];
                ci++;
            }
            break;
        }
        else if(bi >= b_size) {
            for(int i = ai; i < a_size; i++) {
                c[ci] = a[i];
                ci++;
            }
            break;
        }
    }
};
//agg should start at 2
void recursive_merge(double (&a)[], double (&b)[], int taskid, int offset, int rows, int aggregate) {
// CALI_CXX_MARK_FUNCTION;
    int mtype = FROM_WORKER;
    //if process is a worker at this current level proceed
    if(taskid%aggregate == 1) {
        //store current info before overwriting
        int moffset = offset;
        int mrows = rows;
        //receive info
        CALI_MARK_BEGIN(comm);
        CALI_MARK_BEGIN(comm_small);
        CALI_MARK_BEGIN(MPI_Recv_Region);
        MPI_Recv(&offset, 1, MPI_INT, taskid + (aggregate/2), mtype, MPI_COMM_WORLD, &status);
        CALI_MARK_END(MPI_Recv_Region);
        CALI_MARK_BEGIN(MPI_Recv_Region);
        MPI_Recv(&rows, 1, MPI_INT, taskid + (aggregate/2), mtype, MPI_COMM_WORLD, &status);
        CALI_MARK_END(MPI_Recv_Region);
        CALI_MARK_END(comm_small);
        CALI_MARK_BEGIN(comm_large);
        CALI_MARK_BEGIN(MPI_Recv_Region);
        MPI_Recv(&a[mrows], rows, MPI_DOUBLE, taskid + (aggregate/2), mtype, MPI_COMM_WORLD, &status);
        CALI_MARK_END(MPI_Recv_Region);
        CALI_MARK_END(comm_large);
        CALI_MARK_END(comm);
        // for(int i = 0; i < rows + mrows; i++) {
        //     printf("task: %d - at %d: %f\n", taskid, i, a[i]);
        // }

        int off = moffset > offset ? offset : moffset;
        
        //merge incoming with current (leaves offset as moffset in b with length mrows + rows)
        CALI_MARK_BEGIN(comp);
        CALI_MARK_BEGIN(comp_large);
        // merge(a, b, b, offset, rows, moffset, mrows, off);
        merge(a, b, 0, mrows, rows + mrows);
        CALI_MARK_END(comp_large);
        CALI_MARK_END(comp);
        
        //recurse a level up
        if(rows + mrows != sizeOfMatrix) {
            recursive_merge(a,b, taskid, off, mrows + rows, aggregate*2);
        }
        else {
            MPI_Send(&a, sizeOfMatrix, MPI_DOUBLE, MASTER, mtype, MPI_COMM_WORLD);
        }
    }
    //only half the processes are working, so the other half must send their data to the worker
    else {
        CALI_MARK_BEGIN(comm);
        CALI_MARK_BEGIN(comm_small);
        CALI_MARK_BEGIN(MPI_Send_Region);
        MPI_Send(&offset, 1, MPI_INT, taskid - (aggregate/2), mtype, MPI_COMM_WORLD);
        CALI_MARK_END(MPI_Send_Region);
        CALI_MARK_BEGIN(MPI_Send_Region);
        MPI_Send(&rows, 1, MPI_INT, taskid - (aggregate/2), mtype, MPI_COMM_WORLD);
        CALI_MARK_END(MPI_Send_Region);
        CALI_MARK_END(comm_small);
        CALI_MARK_BEGIN(comm_large);
        CALI_MARK_BEGIN(MPI_Send_Region);
        MPI_Send(&(a), rows, MPI_DOUBLE, taskid - (aggregate/2), mtype, MPI_COMM_WORLD);
        CALI_MARK_END(MPI_Send_Region);
        CALI_MARK_END(comm_large);
        CALI_MARK_END(comm);
    }
};

void fill_data(double (&arr)[], char data_order, int sizeOfMatrix) {
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


int main (int argc, char *argv[])
{
CALI_CXX_MARK_FUNCTION;

sizeOfMatrix = atoi(argv[1]);
data_order = *(argv[2]);

int	numtasks,              /* number of tasks in partition */
	taskid,                /* a task identifier */
	numworkers,            /* number of worker tasks */
	source,                /* task id of message source */
	dest,                  /* task id of message destination */
	mtype,                 /* message type */
	rows,                  /* rows of matrix A sent to each worker */
	averow, extra, offset; /* used to determine rows sent to each worker */
double	a[sizeOfMatrix],   /* receiving array*/
	    b[sizeOfMatrix];       /* whole array */
int i, j, k, rc;           /* misc */

MPI_Init(&argc,&argv);
MPI_Comm_rank(MPI_COMM_WORLD,&taskid);
MPI_Comm_size(MPI_COMM_WORLD,&numtasks);
if (numtasks < 2 ) {
  printf("Need at least two MPI tasks. Quitting...\n");
  MPI_Abort(MPI_COMM_WORLD, rc);
  exit(1);
  }
numworkers = numtasks-1;

cali::ConfigManager mgr;
mgr.start(); 
    
    if (taskid == MASTER)
    {

        // INITIALIZATION PART FOR THE MASTER PROCESS STARTS HERE

        printf("merge sort has started with %d tasks.\n",numtasks);
        printf("Initializing arrays...\n");
        CALI_MARK_BEGIN(data_init);
        fill_data(b,data_order,sizeOfMatrix);
        for(int i = 0; i < sizeOfMatrix; i++) {
            a[i] = b[i];
        }
        CALI_MARK_END(data_init);

        // for(int i = 0; i < sizeOfMatrix; i++) {
        //     printf("b at %d: %f\n", i, b[i]);
        // }small

        averow = sizeOfMatrix/numworkers;
        extra = sizeOfMatrix%numworkers;
        offset = 0;
        mtype = FROM_MASTER;
        for(int i = 1; i <= numworkers; i++) {
            rows = (i <= extra) ? averow+1 : averow;   	
            // printf("Sending %d rows to task %d offset=%d\n",rows,i,offset);
            CALI_MARK_BEGIN(comm);
            CALI_MARK_BEGIN(comm_small);
            CALI_MARK_BEGIN(MPI_Send_Region);
            MPI_Send(&offset, 1, MPI_INT, i, mtype, MPI_COMM_WORLD);
            CALI_MARK_END(MPI_Send_Region);
            CALI_MARK_BEGIN(MPI_Send_Region);
            MPI_Send(&rows, 1, MPI_INT, i, mtype, MPI_COMM_WORLD);
            CALI_MARK_END(MPI_Send_Region);
            CALI_MARK_END(comm_small);
            CALI_MARK_BEGIN(comm_large);
            CALI_MARK_BEGIN(MPI_Send_Region);
            MPI_Send(&(a[offset]), rows, MPI_DOUBLE, i, mtype,
                    MPI_COMM_WORLD);
            CALI_MARK_END(MPI_Send_Region);
            CALI_MARK_END(comm_large);
            CALI_MARK_END(comm);
            offset += rows;
        }
        CALI_MARK_BEGIN(comm);
        CALI_MARK_BEGIN(comm_large);
        CALI_MARK_BEGIN(MPI_Recv_Region);
        MPI_Recv(&a, sizeOfMatrix, MPI_DOUBLE, 1, FROM_WORKER, MPI_COMM_WORLD, &status);
        CALI_MARK_END(MPI_Recv_Region);
        CALI_MARK_END(comm_large);
        CALI_MARK_END(comm);
    }

    if(taskid > MASTER) {
        //receive info from master
        mtype = FROM_MASTER;
        CALI_MARK_BEGIN(comm);
        CALI_MARK_BEGIN(comm_small);
        CALI_MARK_BEGIN(MPI_Recv_Region);
        MPI_Recv(&offset, 1, MPI_INT, MASTER, mtype, MPI_COMM_WORLD, &status);
        CALI_MARK_END(MPI_Recv_Region);
        CALI_MARK_BEGIN(MPI_Recv_Region);
        MPI_Recv(&rows, 1, MPI_INT, MASTER, mtype, MPI_COMM_WORLD, &status);
        CALI_MARK_END(MPI_Recv_Region);
        CALI_MARK_END(comm_small);
        CALI_MARK_BEGIN(comm_large);
        CALI_MARK_BEGIN(MPI_Recv_Region);
        MPI_Recv(&a, rows, MPI_DOUBLE, MASTER, mtype, MPI_COMM_WORLD, &status);
        CALI_MARK_END(MPI_Recv_Region);
        CALI_MARK_END(comm_large);
        CALI_MARK_END(comm);
        //sort dataset
        CALI_MARK_BEGIN(comp);
        CALI_MARK_BEGIN(comp_large);
        qsort((void*)&(a), rows, sizeof(double), compInt);
        CALI_MARK_END(comp_large);
        CALI_MARK_END(comp);
        //merge up
        recursive_merge(a,b,taskid,offset,rows,2);
    }

    if(taskid == MASTER) {
        //check array order
        bool err_found = false;
        CALI_MARK_BEGIN(correctness_check);
        for (int i = 1; i < sizeOfMatrix; i++) {
            if (a[i] < a[i - 1]) {
                printf("Error. Out of order sequence: %d found at: %d after value: %d\n", a[i], i, a[i-1]);
                err_found = true;
                break;
            }
        }
        if(err_found) {
            for(int i = 0; i < sizeOfMatrix; i++) {
                printf("b at %d: %f\n", i, a[i]);
            }
        }
        else {
            printf("Array is properly sorted\n");
        }
        CALI_MARK_END(correctness_check);
    }

    adiak::init(NULL);
    adiak::launchdate();    // launch date of the job
    adiak::libraries();     // Libraries used
    adiak::cmdline();       // Command line used to launch the job
    adiak::clustername();   // Name of the cluster
    adiak::value("Algorithm", "MergeSort"); // The name of the algorithm you are using (e.g., "MergeSort", "BitonicSort")
    adiak::value("ProgrammingModel", "MPI"); // e.g., "MPI", "CUDA", "MPIwithCUDA"
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
    }
    adiak::value("num_procs", numtasks); // The number of processors (MPI ranks)
    adiak::value("num_threads", numworkers); // The number of CUDA or OpenMP threads
    adiak::value("num_blocks", 0); // The number of CUDA blocks 
    adiak::value("group_num", 25); // The number of your group (integer, e.g., 1, 10)
    adiak::value("implementation_source", "Handwritten"); // Where you got the source code of your algorithm; choices: ("Online", "AI", "Handwritten").

   // Flush Caliper output before finalizing MPI
   mgr.stop();
   mgr.flush();

   MPI_Finalize();

   return 0;
};
