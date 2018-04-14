// matrix vector multiplication with parallel reduction
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <iostream>
#include <chrono>
#include "helper/wtime.h"
using namespace std;


// rows = M = 10240, columns = N = 256, vector b[256].  1 block operates on 1 row. Each block has 256 threads.
// Number of blocks = 128, so increment tid by 128.
__global__ void prefix_scan (int *x, int *x_d int n) { 
  	int tid = threadIdx.x + blockIdx.x * blockDim.x;		// initialize with block number. Tid = 0 -> 10240
 	//__shared__ has scope of block. All threads in block has access to it.
 	__shared__ int smem[8];   
 	//copy vector ‘b’ element to a corresponding thread location index
 	smem[tid] = x[tid];
 	__syncthreads(); 	//wait for all threads

 while (tid < n) {  		
     // Each thread in a block will operate on 1 element from row and 1 element from vector
     // 256 threads in a block, and 256 columns as well as vector array elements = 256
     // results are stored in smem[0->255]
     x_d[tid] += smem[tid-1];
     __syncthreads();

     // Threads should also perform parallel reduction to sum up smem[256] vector.
     // Block dimension is reduced to half after every for loop execution.

     tid += n;	// Jump to next block which is away by 128 blocks w.r.t. current one
 } // end while
} // end kernel function


int
main (int args, char **argv)
{
  int n;
  //n = 32000000;
  n = 8;	   
  int *x= (int *)malloc(sizeof(int)*n);
	
  cout << "original array: ";
  for (int i = 0; i < n; i++) {     
	  x[i] = rand () % 5;  
	  cout << x[i] << " ";       
  }   cout << endl;
	
  int *x_d;	//device storage pointers 

  cudaMalloc ((void **) &x_d, sizeof (int) * n);

  cudaMemcpy (x_d, x, sizeof (int) * n, cudaMemcpyHostToDevice);
  
  // perform prefix_scan on GPU
  auto time_beg = wtime();  
  prefix_scan <<< 128,128 >>> (x_d,n );
  cudaMemcpy (x, x_d, sizeof (int) * n, cudaMemcpyDeviceToHost);
  auto el = wtime() - time_beg;
 // cout << "Time for <128,128> is: " << el << " Sec " << endl;

  cout << "result is: " ;
	for (int i = 0; i < n; i++){
		cout << x_d[i] << " ";
	}
	cout << endl;
	
    return 0;
}
