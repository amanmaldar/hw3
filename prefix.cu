// matrix vector multiplication with parallel reduction
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <iostream>
#include <chrono>
#include "helper/wtime.h"
using namespace std;
__global__ void vec_mult_kernel (int *a, int *b, int *c, int M, int N) {
int tid = blockIdx.x; // initialize with block number. Tid = 0 -> 10240
__shared__ int smem[256];
smem[threadIdx.x] = b[threadIdx.x];
__syncthreads(); //wait for all threads
while (tid < M) {
smem[threadIdx.x] *= a[tid*N + threadIdx.x];
__syncthreads();
// Threads should also perform parallel reduction to sum up smem[256] vector
// size is also reduced to half after every for loop execution.
for (int i= blockDim.x/2; i>0; i = i/2) {
if(threadIdx.x < i) {
int temp = smem[threadIdx.x] + smem[threadIdx.x + i];
smem[threadIdx.x] = temp;
}
} // for ends
// once parallel reduction is complete, result is at smem[0]
if (threadIdx.x == 0) {
c[tid] = smem[0];
}
tid += 128; // Jump to next block which is away by 128 blocks w.r.t. current one
} // end while
} // end kernel function

int
main (int args, char **argv)
{
// configure matrix dimensions
int M,N,P;
M = 10240;
N = 256;
P = 1;
int *a= (int *)malloc(sizeof(int)*M*N);
int *b= (int *)malloc(sizeof(int)*N*P);
int *c= (int *)malloc(sizeof(long)*M*P);
// Initialize matrix A and B
for (int i = 0; i < M*N; i++) { a[i] = rand () % 5; }
for (int i = 0; i < N*P; i++) { b[i] = rand () % 5; }
int *a_d, *b_d, *c_d; //device storage pointers
cudaMalloc ((void **) &a_d, sizeof (int) * M*N);
cudaMalloc ((void **) &b_d, sizeof (int) * N*P);
cudaMalloc ((void **) &c_d, sizeof (int) * M*P);
cudaMemcpy (a_d, a, sizeof (int) * M*N, cudaMemcpyHostToDevice);
cudaMemcpy (b_d, b, sizeof (int) * N*P, cudaMemcpyHostToDevice);
// perform multiplication on GPU
auto time_beg = wtime();
vec_mult_kernel <<< 128,256 >>> (a_d, b_d, c_d, M, N );
cudaMemcpy (c, c_d, sizeof (int) * M*P, cudaMemcpyDeviceToHost);
auto el = wtime() - time_beg;
cout << "Time for <128,256> is: " << el << " Sec " << endl;
return 0;
}
