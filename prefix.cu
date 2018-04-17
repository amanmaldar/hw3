// limited to 16284 enetries. 128 x 128 kernel is used. 16384 elements are copied to memory and each block performs klogg alogo on 128 
// elements . results are pushed back to cpu and cpu performs the final addition.
// question - i want to have more than 128 x 128 elements. lets say 128 x 128 x 4. we will see how kernel performs in next program
// assert ref : https://stackoverflow.com/questions/3767869/adding-message-to-assert?utm_medium=organic&utm_source=google_rich_qa&utm_campaign=google_rich_qa
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <iostream>
#include <chrono>
#include <math.h>
#include "helper/wtime.h"
using namespace std;


#ifndef NDEBUG
#   define ASSERT(condition, message) \
    do { \
        if (! (condition)) { \
            std::cerr << "Assertion `" #condition "` failed in " << __FILE__ \
                      << " line " << __LINE__ << ": " << message << std::endl; \
            std::terminate(); \
        } \
    } while (false)
#else
#   define ASSERT(condition, message) do { } while (false)
#endif

void fillPrefixSum(int arr[], int n, int prefixSum[])
{
    prefixSum[0] = arr[0];
    // Adding present element with previous element
    for (int i = 1; i < n; i++)
        prefixSum[i] = prefixSum[i-1] + arr[i];
}

__device__ int res=0;  //result from one block to next block
__device__ int smem[32000000]; // 128*128 
__device__ int tmp1;
      

__global__ void vec_mult_kernel (int *b_d, int *a_d, int n, int depth) {
  
int tid = blockIdx.x* blockDim.x+ threadIdx.x; 
    


  int d = 0;
  int offset = 0;
    
  while (tid < n) {
      smem[tid] = a_d[tid];   // copy data to shared memory
  __syncthreads(); //wait for all threads

  if (threadIdx.x == 0 ) {   b_d[tid] = smem[tid]; }  

  offset = 1; //1->2->4
  for (d =0; d < depth ; d++){                        // depth = 3
    
    if (tid%blockDim.x >= offset){  
     
      smem[tid] += smem[tid-offset] ;           //after writing to smem do synchronize
      __syncthreads();      
       
    }// end if
    offset *=2;
   } // end for 
         b_d[tid] = smem[tid]; 

      __syncthreads();
     // 3 new line below
     
      if (blockIdx.x != 0 && threadIdx.x == 0) 
          //if ( threadIdx.x == 0) 
      {
          tmp1 = smem[tid-1]; __syncthreads();
          smem[tid] += tmp1; __syncthreads();
      }
      else if( blockIdx.x != 0 && threadIdx.x > 0 && threadIdx.x < 4)
      {
        smem[tid]+= tmp1; __syncthreads();
      }
      
      b_d[tid] = smem[tid]; 
   
      //b_d[tid] = tid; 
      __syncthreads();
  
      tid += 4; //there are no actual grid present, we just increment the tid to fetch next elemennts from input array
} // end while (tid < n)
} // end kernel function


int
main (int args, char **argv)
{
  int threadsInBlock = 4;
  int numberOfBlocks = 4;
  //int n = threadsInBlock*numberOfBlocks;
  int n = 8;
  //int b_cpu[n];
  int depth = log2(threadsInBlock);    //log(blockDim.x) = log(8) = 3,  blockDim.x = threadsInBlock

  int *a_cpu= (int *)malloc(sizeof(int)*n);
  int *b_cpu= (int *)malloc(sizeof(int)*n);
  int *b_ref= (int *)malloc(sizeof(int)*n);
    
  cout << "\n array is: "; 
  for (int i = 0; i < n; i++) { a_cpu[i] = rand () % 5 + 2; cout << a_cpu[i] << " ";
                              }   cout << endl;
  
  auto time_beg = wtime();
  fillPrefixSum(a_cpu, n, b_ref);
  auto el_cpu = wtime() - time_beg;
  
  cout << "\n CPU Result is: "; 
  for (int i = 0; i < n; i++) 
  { cout << b_ref[i] << " ";   
  } cout << endl;
  
  int *a_d, *b_d; //device storage pointers

  cudaMalloc ((void **) &a_d, sizeof (int) * n);
  cudaMalloc ((void **) &b_d, sizeof (int) * n);

  cudaMemcpy (a_d, a_cpu, sizeof (int) * n, cudaMemcpyHostToDevice);

  time_beg = wtime();
  vec_mult_kernel <<< numberOfBlocks,threadsInBlock >>> (b_d,a_d, n, depth );
  cudaMemcpy (b_cpu, b_d, sizeof (int) * n, cudaMemcpyDeviceToHost);


    // cpu combines the results of each block with next block. cpu basically adds last element from previos block to
    // next element in next block. This is sequential process.
  /*  int res = 0;
    for (int i=0;i<n;i++){
         b_cpu[i]+=res;
        if((i+1)%threadsInBlock==0){ res = b_cpu[i]; }        
    }*/
      auto el_gpu = wtime() - time_beg;


  cout << "\n GPU Result is: ";
  for (int i = 0; i < n; i++) {    
   // ASSERT(b_ref[i]== b_cpu[i], "Error at i= " << i << 
     //     " b_ref[i]: " << b_ref[i] << " b_cpu[i]: " << b_cpu[i] <<
      //    " b_ref[i+1]: " << b_ref[i+1] << " b_cpu[i+1]: " << b_cpu[i+1] << 
       //   " b_ref[i+2]: " << b_ref[i+2] <<  " b_cpu[i+2]: " << b_cpu[i+2] << 
        //  " a_cpu[i+1]: " << a_cpu[i+1] << " a_cpu[i+2]: " << a_cpu[i+2] );  
      //ASSERT(b_ref[i] == b_cpu[i], "Error at i= " << i);  
    cout << b_cpu[i] << " ";  
  } cout << endl;

  cout << "CPU time is: " << el_cpu * 1000 << " mSec " << endl;
  cout << "GPU time is: " << el_gpu * 1000 << " mSec " << endl; 
  return 0; //new
}
