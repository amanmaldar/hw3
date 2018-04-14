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
__global__ void vec_mult_kernel (int *a, int *b, int *c, int M, int N) { 
  	int tid = blockIdx.x;		// initialize with block number. Tid = 0 -> 10240
 	//__shared__ has scope of block. All threads in block has access to it.
 	__shared__ int smem[256];   
 	//copy vector ‘b’ element to a corresponding thread location index
 	smem[threadIdx.x] = b[threadIdx.x];
 	__syncthreads(); 	//wait for all threads

 while (tid < M) {  		
     // Each thread in a block will operate on 1 element from row and 1 element from vector
     // 256 threads in a block, and 256 columns as well as vector array elements = 256
     // results are stored in smem[0->255]
     smem[threadIdx.x] *= a[tid*N + threadIdx.x];
     __syncthreads();

     // Threads should also perform parallel reduction to sum up smem[256] vector.
     // Block dimension is reduced to half after every for loop execution.
	
   //parallel reduction
     for (int i= blockDim.x/2; i>0; i = i/2) {
        if(threadIdx.x < i) {
        int temp = smem[threadIdx.x] + smem[threadIdx.x + i];
        smem[threadIdx.x] = temp;
        }
        } // for ends    
     // once parallel reduction is complete, result is at smem[0] in each block.
     if (threadIdx.x == 0) {		//only thread 0 can write the result back. 
       	 c[tid] = smem[0];
     }
         	 tid += 128;	// Jump to next block which is away by 128 blocks w.r.t. current one
 } // end while
} // end kernel function


int
main (int args, char **argv)
{
 // configure matrix dimensions
  int n;
  //n = 32000000;
  n = 8;	   
  int *x= (int *)malloc(sizeof(int)*n);

  // Initialize matrix A and B
  cout << "original array: ";
  for (int i = 0; i < n; i++) {     
	  x[i] = rand () % 5;  
	  cout << x[i] << " ";       
  }
  cout << endl;
	
  int *x_d;	//device storage pointers 

  cudaMalloc ((void **) &x_d, sizeof (int) * n);


  cudaMemcpy (x_d, x, sizeof (int) * n, cudaMemcpyHostToDevice);
  
  // perform multiplication on GPU
  auto time_beg = wtime();  
 // vec_mult_kernel <<< 128,128 >>> (a_d, b_d, c_d, M, N,P );
  cudaMemcpy (x, x_d, sizeof (int) * n, cudaMemcpyDeviceToHost);
  auto el = wtime() - time_beg;
  cout << "Time for <128,128> is: " << el << " Sec " << endl;

  cout << "result is: " ;
	for (int i = 0; i < n; i++){
		cout << x[i] << " ";
	}
	cout << endl;
	
    return 0;
}
