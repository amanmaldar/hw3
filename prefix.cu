// matrix vector multiplication with parallel reduction
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <iostream>
#include <chrono>
#include "helper/wtime.h"
using namespace std;
__global__ void vec_mult_kernel (int *a, int n) {
int tid = blockIdx.x; // initialize with block number. Tid = 0 -> 10240
__shared__ int smem[256];
smem[threadIdx.x] = a[threadIdx.x];
__syncthreads(); //wait for all threads
while (tid < n) {
  if (tid == 0) { a[0] = a[1];}
//smem[threadIdx.x] *= a[tid*N + threadIdx.x];
smem[threadIdx.x] += smem[threadIdx.x - 1] ;
  a[threadIdx.x] = smem[threadIdx.x];
__syncthreads();
tid += 128; // Jump to next block which is away by 128 blocks w.r.t. current one
} // end while
} // end kernel function

int
main (int args, char **argv)
{
// configure matrix dimensions
int n = 8;
int *a= (int *)malloc(sizeof(int)*n);
// Initialize matrix A and B
  cout << "array is: ";
for (int i = 0; i < n; i++) { a[i] = rand () % 5 + 2; cout << a[i] << " ";}
  cout << endl;
int *a_d; //device storage pointers

cudaMalloc ((void **) &a_d, sizeof (int) * n);
cudaMemcpy (a_d, a, sizeof (int) * n, cudaMemcpyHostToDevice);

// perform multiplication on GPU
auto time_beg = wtime();
vec_mult_kernel <<< 128,256 >>> (a_d, n );
cudaMemcpy (a, a_d, sizeof (int) * n, cudaMemcpyDeviceToHost);
  cout << "result is: ";
for (int i = 0; i < n; i++) {  cout << a[i] << " ";}
  cout << endl;
auto el = wtime() - time_beg;
cout << "Time for <128,256> is: " << el << " Sec " << endl;
return 0;
}
