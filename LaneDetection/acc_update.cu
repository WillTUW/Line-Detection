#include "acc_update.cuh"

//Can you use functions in cuda? Like round?
__global__ void GPU_UpdateAccumulator(int i, int j, int numrho, short* adata, int* max_val, int* max_n)
{
	// update accumulator, find the most probable line
	//for (int n = 0; n < NUM_ANGLE; n++, adata += numrho)
	int n = threadIdx.x;
	if (n >= NUM_ANGLE)
	{
		//wot
		return; //No no you right we want to use return to get out of this func :thumbsup:
	}

	__shared__ int smax_val[NUM_ANGLE];
	__shared__ int smax_n[NUM_ANGLE];

	int r = round(j * hough_cos(n) + i * hough_sin(n)) + ((numrho - 1) / 2);

	adata[r + (n * numrho)] += 1;
	int val = adata[r + (n * numrho)];

	smax_val[n] = val;
	smax_n[n] = n;
	__syncthreads();

	for (int s = 1; s < NUM_ANGLE; s *= 2)
	{
		int index = (2 * s) * n; // Next
		if (index < NUM_ANGLE)
		{
			if (smax_val[index + s] > smax_val[index])
			{
				smax_val[index] = smax_val[index + s];
				smax_n[index] = smax_n[index + s];
			}
		}

		__syncthreads();
	}

	if (n == 0)
	{
		*max_val = smax_val[0];
		*max_n = smax_n[0];
	}
}

void UpdateAccumulator(int i, int j, int numrho, short* dev_adata, int* dev_max_val, int* dev_max_n, 
	short* adata, int *max_val, int *max_n, cudaEvent_t cuEvent, cudaStream_t stream1, cudaStream_t stream2)
{
	int host_max_val[1];
	int host_max_n[1];

	// Copy input vectors from host memory to GPU buffers.

	// old mem copy
	// cudaMemcpy(dev_adata, adata, NUM_ANGLE * numrho * sizeof(short), cudaMemcpyHostToDevice);

	//*
	// Async mem copy
	cudaMemcpyAsync(dev_adata, adata, NUM_ANGLE * numrho * sizeof(short), cudaMemcpyHostToDevice, stream1);

	// sync point
	cudaEventRecord(cuEvent, stream1); // record event
	cudaStreamWaitEvent(stream2, cuEvent, 0); // wait for event in stream1
	//*/

	GPU_UpdateAccumulator <<< 1, NUM_ANGLE, 1, stream2>>> (i, j, numrho, dev_adata, dev_max_val, dev_max_n);

	/*
	// old mem cpy
	cudaDeviceSynchronize();
	
	cudaMemcpy(adata, dev_adata, NUM_ANGLE * numrho * sizeof(short), cudaMemcpyDeviceToHost);
	cudaMemcpy(host_max_val, dev_max_val, 1 * sizeof(int), cudaMemcpyDeviceToHost);
	cudaMemcpy(host_max_n, dev_max_n, 1 * sizeof(int), cudaMemcpyDeviceToHost);
	//*/

	//*
	// Async mem copy
	cudaMemcpyAsync(adata, dev_adata, NUM_ANGLE * numrho * sizeof(short), cudaMemcpyDeviceToHost, stream1);
	cudaMemcpyAsync(host_max_val, dev_max_val, 1 * sizeof(int), cudaMemcpyDeviceToHost, stream1);
	cudaMemcpyAsync(host_max_n, dev_max_n, 1 * sizeof(int), cudaMemcpyDeviceToHost, stream1);
	//*/

	*max_val = host_max_val[0];
	*max_n = host_max_n[0];
}
