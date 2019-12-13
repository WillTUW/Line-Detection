#pragma once
#include "acc_update.cuh"

#define THRESHOLD 80
#define LINE_GAP 50
#define LINE_LENGTH 150

__global__ void GPU_Hough(int width, int height, int *queueX, int *queueY, int count,
	int *adata, unsigned char* maskData, int numrho, int* outX0, int *outY0, int *outX1, int *outY1);

void Hough(int width, int height, int *queueX, int *queueY, int count,
	int *adata, unsigned char* maskData, int numrho, int* outX0, int *outY0, int *outX1, int *outY1);