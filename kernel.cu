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
	//clock_t start, end;
	//struct timespec start, end;

	// Metrics
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	cudaError_t cudaStatus;
	cudaDeviceProp prop;

	baseIntervalo = 1.0 / (double)cantidadIntervalos;
	double totalSum = 0;

	//Declare variables
	cudaGetDeviceProperties(&prop, 0);
	int num_threads_supported = prop.maxThreadsPerBlock;
	int num_blocks_supported = prop.maxThreadsDim[0];
	
	
	//Grid Size
	
	int NUM_BLOCKS = num_blocks_supported;

	//Threadblock size
	int NUM_THREADS = num_threads_supported;
	long long total_threads = NUM_BLOCKS * NUM_THREADS;

	double* arr;
	//int size = total_threads * sizeof(double);

	//dynamic allocation
	cudaMallocManaged(&arr, total_threads * sizeof(double));
	// Initialize array in device to 0
	cudaMemset(arr, 0, total_threads * sizeof(double));
	// Launch Kernel
	cudaEventRecord(start);
	cudaEventRecord(stop);

	cudaEventRecord(start);
	//Launch the kernel
	kernel << < NUM_BLOCKS, NUM_THREADS >> > (arr, total_threads, baseIntervalo, cantidadIntervalos);

	
	cudaEventRecord(stop);

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
	cudaEventSynchronize(stop);
	float milliseconds = 0;
	cudaEventElapsedTime(&milliseconds, start, stop);
	


	for (long long c = 0; c < total_threads; c++)
	{
		totalSum += arr[c];
	}

Error:
	//De-allocate memory
	cudaFree(arr);

	//double total = (double)(end.tv_sec - start.tv_sec) + ((double)(end.tv_nsec - start.tv_nsec) / 1000000000L);
	//printf("Result = %20.18lf (%.10lf ms)\n\n", totalSum, total*1000);
	printf("PI = %.10f (Total time: %lf)\n", totalSum, milliseconds);
	return 0;
}
