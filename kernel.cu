#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include <Windows.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <time.h>


long long cantidadIntervalos = 1000000000; // 1 B
double baseIntervalo;

__global__ void kernel(double* d_a, long long total_threads, double baseIntervalo, long long cantidadIntervalos)
{
	//calculate global thread ID(tid)
	long long tid = (blockIdx.x * blockDim.x) + threadIdx.x;
	double acum = 0;
	double fdx = 0;
	double x = 0;

	if (tid < total_threads)
	{
		for (long long i = tid; i < cantidadIntervalos; i += total_threads)
		{
			x = (i+0.5) * baseIntervalo;
			fdx = 4 / (1 + x * x);
			acum += fdx;
		}
		acum *= baseIntervalo;
		d_a[tid] = acum;
	}
}

int main(void)
{
	clock_t start, end;
	cudaError_t cudaStatus;
	cudaDeviceProp prop;

	baseIntervalo = 1.0 / (double)cantidadIntervalos;
	double totalSum = 0;

	//Declare variables
	cudaGetDeviceProperties(&prop, 0);
	int num_threads_supported = prop.maxThreadsPerBlock;
	int num_blocks_supported = prop.maxThreadsDim[0];
	
	
	//Grid Size
		//int NUM_BLOCKS = (int)ceil(datos / NUM_THREADS);
	int NUM_BLOCKS = num_blocks_supported;

	//Threadblock size
	int NUM_THREADS = num_threads_supported;
	long long total_threads = NUM_BLOCKS * NUM_THREADS;


	//array that will contain partial sums of each thread
	double* h_a = (double*)malloc(total_threads *sizeof(double));

	//Allocate arrays in device memory, array that will contain partial sums of each thread
	double* d_a;

	//Allocate memory on the device
	cudaStatus = cudaMalloc((void**)&d_a, total_threads * sizeof(double));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	//copy memory from host to Device
	cudaStatus = cudaMemcpy(d_a, h_a, total_threads * sizeof(double), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}


	start = clock();
	//Launch the kernel
	kernel << < NUM_BLOCKS, NUM_THREADS >> > (d_a, total_threads, baseIntervalo, cantidadIntervalos);
	end = clock();


	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
		goto Error;
	}

	//cudaDeviceSynchronize waits for the kernel to finish, and returns
   // any errors encountered during the launch.
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
		goto Error;
	}

	// Copy data back to host
	cudaStatus = cudaMemcpy(h_a, d_a, total_threads * sizeof(double), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

Error:
	//De-allocate memory
	cudaFree(d_a);

	for (long long c = 0; c < total_threads; c++)
	{
		totalSum += h_a[c];
	}

	time_t total = (end - start);
	printf("Result = %20.18lf (%lld)\n\n", totalSum, total);



	free(h_a);





	return 0;
}
