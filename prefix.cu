// matrix vector multiplication with parallel reduction
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <iostream>
#include <chrono>
#include "helper/wtime.h"
using namespace std;
__global__ void vec_mult_kernel (int *a, int n) {
int tid = threadIdx.x; // initialize with block number. Tid = 0 -> 10240
__shared__ int smem[256];
smem[threadIdx.x] = a[threadIdx.x];
__syncthreads(); //wait for all threads
while (tid < n) {
  if (tid == 0) { smem[0] = a[0]; a[threadIdx.x] = smem[threadIdx.x];  break;}

  for (int depth = 0; depth < 3; depth++)

  smem[threadIdx.x] += a[threadIdx.x-1] ;
  a[threadIdx.x] = smem[threadIdx.x];
__syncthreads();
tid += 128; // Jump to next block which is away by 128 blocks w.r.t. current one
} // end while
} // end kernel function




__global__ void scan(int *g_odata, int *g_idata, int n)
{
 extern __shared__ float temp[]; // allocated on invocation
 int thid = threadIdx.x;
 int pout = 0, pin = 1;
 // load input into shared memory.
 // This is exclusive scan, so shift right by one and set first elt to 0
 temp[pout*n + thid] = (thid > 0) ? g_idata[thid-1] : 0;
 __syncthreads();
 for (int offset = 1; offset < n; offset *= 2)
 {
 pout = 1 - pout; // swap double buffer indices
 pin = 1 - pout;
 if (thid >= offset)
 temp[pout*n+thid] += temp[pin*n+thid - offset];
 else
 temp[pout*n+thid] = temp[pin*n+thid];
 __syncthreads();
 }
 g_odata[thid] = temp[pout*n+thid1]; // write output
} 



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
//vec_mult_kernel <<< 128,256 >>> (a_d, n );
void scan(int *b_d, int *a_d, int n)
cudaMemcpy (b, b_d, sizeof (int) * n, cudaMemcpyDeviceToHost);
  cout << "result is: ";
for (int i = 0; i < n; i++) {  cout << b[i] << " ";}
  cout << endl;
auto el = wtime() - time_beg;
cout << "Time for <128,256> is: " << el << " Sec " << endl;
return 0;
}
