
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <iostream>
#include <chrono>
#include <math.h>
#include "helper/wtime.h"
using namespace std;

void fillPrefixSum(int arr[], int n, int prefixSum[])
{
    prefixSum[0] = arr[0];
    // Adding present element with previous element
    for (int i = 1; i < n; i++)
        prefixSum[i] = prefixSum[i-1] + arr[i];
}

__device__ int res=0;  //result from one block to next block


__global__ void vec_mult_kernel (int *b_d, int *a_d, int n, int depth) {
  
int tid = blockIdx.x* blockDim.x+ threadIdx.x; 
 // int smemSize = blockDim.x*gridDim.x;
__shared__ int smem[128];    // numberOfBlocks*threadsInBlock  = 2^7 + 2^7 = 16K shared memory
  int d = 0;
  int offset = 0;
  
  while (tid < n) {
  smem[threadIdx.x] = a_d[blockIdx.x*128+threadIdx.x];   // copy data to shared memory
  __syncthreads(); //wait for all threads

  if (tid%128 == 0 ) { smem[0] = smem[0] + res;   //** add previos result to telement zero
                              b_d[blockIdx.x*128] = smem[0]; break;}  

  offset = 1; //1->2->4
  for (d =0; d < 7 ; d++){                        // depth = 3
    
    if (tid%blockDim.x >= offset){  
     
      smem[threadIdx.x] += smem[threadIdx.x-offset] ;           //after writing to smem do synchronize
      __syncthreads();      
       
    }// end if
    offset *=2;
   } // end for 
   b_d[blockIdx.x*128+threadIdx.x] = smem[threadIdx.x]; //+ res; no need as we alreasy are adding result to element zero above **  // save result to b_d after adding res to it;
      __syncthreads();
  if(threadIdx.x == 127) {res = smem[127];     }  // if last thread in block save cout
  tid += 128;

} // end while (tid < n)
} // end kernel function


int
main (int args, char **argv)
{
  int threadsInBlock = 128;
  int numberOfBlocks = 1;
  //int n = threadsInBlock*numberOfBlocks;
  int n = 256;
  int b_cpu[n];
  int depth = log2(threadsInBlock);    //log(blockDim.x) = log(8) = 3,  blockDim.x = threadsInBlock

  int *a= (int *)malloc(sizeof(int)*n);
  int *b= (int *)malloc(sizeof(int)*n);
  
  cout << "\n array is: "; 
  for (int i = 0; i < n; i++) { a[i] = rand () % 5 + 2; //cout << a[i] << " ";
                              }   cout << endl;
  
  auto time_beg = wtime();
  fillPrefixSum(a, n, b_cpu);
  auto el_cpu = wtime() - time_beg;
  
  cout << "\n CPU Result is: "; 
  for (int i = 0; i < n; i++) 
  { cout << b_cpu[i] << " ";   
  } cout << endl;
  
  int *a_d, *b_d; //device storage pointers

  cudaMalloc ((void **) &a_d, sizeof (int) * n);
  cudaMalloc ((void **) &b_d, sizeof (int) * n);

  cudaMemcpy (a_d, a, sizeof (int) * n, cudaMemcpyHostToDevice);

  time_beg = wtime();
  vec_mult_kernel <<< numberOfBlocks,threadsInBlock >>> (b_d,a_d, n, depth );
  auto el_gpu = wtime() - time_beg;
  cudaMemcpy (b, b_d, sizeof (int) * n, cudaMemcpyDeviceToHost);


  cout << "\n GPU Result is: ";
  for (int i = 0; i < n; i++) {    
    //assert(b[i]== b_cpu[i]);   
    cout << b[i] << " ";  
  } cout << endl;

  cout << "CPU time is: " << el_cpu * 1000 << " mSec " << endl;
  cout << "GPU time is: " << el_gpu * 1000 << " mSec " << endl;
  return 0;
}
