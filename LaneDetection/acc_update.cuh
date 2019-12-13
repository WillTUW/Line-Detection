#pragma once
#define SHIFT 16
#define M_PI 3.14159265359
#define M_THETA (M_PI / 180)
#define RHO 1.0
#define IRHO (1 / RHO)
#define hough_cos(x) (cos(x * M_THETA) * IRHO)
#define hough_sin(x) (sin(x * M_THETA) * IRHO)
#define NUM_ANGLE 180

__global__ void GPU_UpdateAccumulator(int i, int j, int numrho, short* adata, int* max_val, int* max_n);
void UpdateAccumulator(int i, int j, int numrho, short* dev_adata, int* dev_max_val, int* dev_max_n, 
	short* adata, int* max_val, int* max_n, cudaEvent_t cuEvent, cudaStream_t stream1, cudaStream_t stream2);

