#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include <Windows.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <time.h>
#include <cuda_fp16.h>

long long cantidadIntervalos = 1000000000; // 1 B
double baseIntervalo;
double acum=0;

__global__ void kernel(long long total_threads, double baseIntervalo, long long cantidadIntervalos, double* totalSum)
{
	//calculate global thread ID(tid)
	long long tid = (blockIdx.x * blockDim.x) + threadIdx.x;
	double localAcum = 0;
	double fdx = 0;
	double x = 0;

	if (tid < total_threads)
	{
		for (long long i = tid; i < cantidadIntervalos; i += total_threads)
		{
			x = (i+0.5) * baseIntervalo;
			fdx = 4 / (1 + x * x);
			localAcum += fdx;
		}
		
		atomicAdd(&totalSum[0], localAcum);		//avoiding race condition
		
		//totalSum[0] += localAcum;			//race condition
	}

}

int main(void)
{
	
	// Metrics
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	cudaError_t cudaStatus;
	cudaDeviceProp prop;

	baseIntervalo = 1.0 / (double)cantidadIntervalos;
	double* totalSum;

	//Declare variables
	cudaGetDeviceProperties(&prop, 0);
	int num_threads_supported = prop.maxThreadsPerBlock;
	int num_blocks_supported = prop.maxThreadsDim[0];


	//Grid Size
	int NUM_BLOCKS = num_blocks_supported;

	//Threadblock size
	int NUM_THREADS = num_threads_supported;
	long long total_threads = NUM_BLOCKS * NUM_THREADS;

	//dynamic allocation
	cudaMallocManaged((void**)&totalSum, sizeof(double));
	// Initialize array in device to 0
	cudaMemset(totalSum, 0, sizeof(double));
	// start recording
	cudaEventRecord(start);

	//Launch the kernel
	kernel << < NUM_BLOCKS, NUM_THREADS >> > (total_threads, baseIntervalo, cantidadIntervalos, totalSum);
		
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

	totalSum[0] *= baseIntervalo;

	printf("PI = %.10f (Total time: %lf)\n", totalSum[0], milliseconds);

Error:
	//De-allocate memory
	cudaFree(totalSum);

	
	return 0;
}
