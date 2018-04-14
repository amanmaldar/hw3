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
__global__ void prefix_scan (int *x_d, int n) { 
  	

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
	  x[i] = rand () % 5 + 2;  
	  cout << x[i] << " ";       
  }   cout << endl;
	
  int *x_d;	//device storage pointers 

  cudaMalloc ((void **) &x_d, sizeof (int) * n);

  cudaMemcpy (x_d, x, sizeof (int) * n, cudaMemcpyHostToDevice);
  
  // perform prefix_scan on GPU
  auto time_beg = wtime();  
  prefix_scan <<< 128,128 >>> (x_d,n);
	cout << "done " ;
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
