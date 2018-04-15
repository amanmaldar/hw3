
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <iostream>
#include <chrono>
#include <math.h>
#include "helper/wtime.h"
using namespace std;
  __device__ int res;  //result from one block to next block


__global__ void vec_mult_kernel (int *b_d, int *a_d, int n, int depth) {
  
int tid = blockIdx.x* blockDim.x+ threadIdx.x; // initialize with block number. Tid = 0 -> 10240
 // int smemSize = blockDim.x*gridDim.x;
__shared__ int smem[128];    // numberOfBlocks*threadsInBlock  = 2^7 + 2^7 = 16K shared memory
  int d = 0;
  int offset = 0;
  
  while (tid < n) {
  smem[tid%128] = a_d[blockIdx.x*127+threadIdx.x];   // copy data to shared memory
  __syncthreads(); //wait for all threads

  if (tid%blockDim.x == 0 ) { smem[tid%128] = a_d[blockIdx.x*127+threadIdx.x];   
                             __syncthreads(); b_d[blockIdx.x*127+threadIdx.x] = smem[tid%128]+res;}

  offset = 1; //1->2->4
  for (d =0; d < depth ; d++){                        // depth = 3
    
    if (tid%blockDim.x >= offset){  
     
      smem[tid%128] += smem[tid%128-offset] ;           //after writing to smem do synchronize
      __syncthreads();      
       
    }// end if
    offset *=2;
   } // end for 
   b_d[blockIdx.x*127+threadIdx.x] = smem[tid%128] + res; // add this part  // save result to b_d after adding res to it;
  if(tid%blockDim.x == blockDim.x-1) {res = b_d[blockIdx.x*127+threadIdx.x];}  // if last thread in block save cout
  __syncthreads();
  tid += blockDim.x;
} // end while (tid < n)
} // end kernel function


void fillPrefixSum(int arr[], int n, int prefixSum[])
{
    prefixSum[0] = arr[0];
    // Adding present element with previous element
    for (int i = 1; i < n; i++)
        prefixSum[i] = prefixSum[i-1] + arr[i];
}



int
main (int args, char **argv)
{
  int threadsInBlock = 2048;
  int numberOfBlocks = 2048;
  int n = threadsInBlock*numberOfBlocks;
  //int n = 16;
  int b_cpu[n];
  int depth = log2(threadsInBlock);    //log(blockDim.x) = log(8) = 3,  blockDim.x = threadsInBlock

  int *a= (int *)malloc(sizeof(int)*n);
  int *b= (int *)malloc(sizeof(int)*n);
  
  cout << "array is: "; 
  for (int i = 0; i < n; i++) { a[i] = rand () % 5 + 2; //cout << a[i] << " ";
                              }   cout << endl;
  
  auto time_beg = wtime();
  fillPrefixSum(a, n, b_cpu);
  auto el_cpu = wtime() - time_beg;
  
  cout << "CPU Result is: "; 
  for (int i = 0; i < n; i++) 
  { //cout << b_cpu[i] << " ";   
  } cout << endl;
  
  int *a_d, *b_d; //device storage pointers

  cudaMalloc ((void **) &a_d, sizeof (int) * n);
  cudaMalloc ((void **) &b_d, sizeof (int) * n);

  cudaMemcpy (a_d, a, sizeof (int) * n, cudaMemcpyHostToDevice);

  time_beg = wtime();
  vec_mult_kernel <<< numberOfBlocks,threadsInBlock >>> (b_d,a_d, n, depth );
  cudaMemcpy (b, b_d, sizeof (int) * n, cudaMemcpyDeviceToHost);
  auto el_gpu = wtime() - time_beg;

  cout << "GPU Result is: ";
  for (int i = 0; i < n; i++) {    
    //assert(b[i]== b_cpu[i]);   
    //cout << b[i] << " ";  
  } cout << endl;

  cout << "CPU time is: " << el_cpu * 1000 << " mSec " << endl;
  cout << "GPU time is: " << el_gpu * 1000 << " mSec " << endl;
  return 0;
}
