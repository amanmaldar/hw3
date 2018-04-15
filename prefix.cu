// matrix vector multiplication with parallel reduction
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <iostream>
#include <chrono>
#include "helper/wtime.h"
using namespace std;
  __device__ int res=0;  //result from one block to next block


__global__ void vec_mult_kernel (int *b_d, int *a_d, int n) {
int tid = blockIdx.x* blockDim.x+ threadIdx.x; // initialize with block number. Tid = 0 -> 10240
__shared__ int smem[256];
  int depth = 3;    //log(blockDim.x) = log(8) = 3
  int d =0;
  int offset = 0;
smem[tid] = a_d[tid];
__syncthreads(); //wait for all threads
while (tid < n) {
  if (tid%blockDim.x == 0 ) { smem[0] = a_d[0]; b_d[0] = smem[0]; tid += 8; break;}
  offset = 1; //1->2->4
  for (d =0; d < depth ; d++){                        // depth = 3
    
    if (tid >= offset){
  
      smem[tid] = smem[tid] + smem[tid-offset] + res ;           //after writing to smem do synchronize
      __syncthreads();
        b_d[tid] = smem[tid];  
        if(tid%blockDim.x == blockDim.x-1) {res = smem[tid];}  // if last thread in block save cout
       
    }// end if
    offset *=2;
   } // end for 
  tid += n;
} // end while (tid < n)
} // end kernel function



int
main (int args, char **argv)
{
// configure matrix dimensions
int n = 8;
int *a= (int *)malloc(sizeof(int)*n);
int *b= (int *)malloc(sizeof(int)*n);
// Initialize matrix A and B
  cout << "array is: ";
for (int i = 0; i < n; i++) { a[i] = rand () % 5 + 2; cout << a[i] << " ";}
  cout << endl;
int *a_d, *b_d; //device storage pointers

cudaMalloc ((void **) &a_d, sizeof (int) * n);
cudaMalloc ((void **) &b_d, sizeof (int) * n);

cudaMemcpy (a_d, a, sizeof (int) * n, cudaMemcpyHostToDevice);

// perform multiplication on GPU
auto time_beg = wtime();
vec_mult_kernel <<< 8,8 >>> (b_d,a_d, n );
cudaMemcpy (b, b_d, sizeof (int) * n, cudaMemcpyDeviceToHost);
  cout << "result is: ";
for (int i = 0; i < n; i++) {  cout << b[i] << " ";}
  cout << endl;
auto el = wtime() - time_beg;
cout << "Time is: " << el << " Sec " << endl;
return 0;
}
