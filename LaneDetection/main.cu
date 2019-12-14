#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cuda.h>
#include <device_functions.h>
#include <cuda_runtime_api.h>
#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>
#include <stdio.h>
#include <conio.h>
#include <stdarg.h>
#include <vector>
#include <algorithm>
#include <chrono>
#include <list>

// Debug constant
constexpr bool DEBUG = true;

// Constants used in hough transform algorithm
constexpr auto SHIFT = 16;
constexpr auto M_PI = 3.14159265359;
#define M_THETA (M_PI / 180)
constexpr auto RHO = 1.0;
#define IRHO (1 / RHO)
#define hough_cos(x) (cos(x * M_THETA) * IRHO)
#define hough_sin(x) (sin(x * M_THETA) * IRHO)
constexpr auto NUM_ANGLE = 180;

// preccentage of file off the top we want to shave off and ignore for lines
constexpr auto T_BOUND_MOD = .6;
constexpr auto BOUND_OFFSET = 22; // offset from image boundry

// parameters for HoughLines
constexpr auto LINE_LENGTH = 200;
constexpr auto LINE_GAP = 10;
constexpr auto LINE_MAX = 20;
constexpr auto LINE_THRESH = 100;

// parameters for preprocessing
constexpr auto KSIZE = 1;

// Final kernel: Updates entire accumulator for all points
__global__ void GPU_UpdateAccumulatorAll(int count, int *queueXY, int numrho, short* adata, short* max_val, short* max_n)
{
	// update accumulator, find the most probable line

	// Use block as index for what point we are processing
	const int point = blockIdx.x;
	if (point >= count)
	{
		return;
	}

	// Extract X and Y of point from compressed integer
	const int j = queueXY[point] >> 16;
	const int i = queueXY[point] & 0xFFFF;

	// Use thread as index for what angle we are processing
	const int n = threadIdx.x;
	if (n >= NUM_ANGLE)
	{
		return;
	}

	// Create shared memory for storing max value and its index
	// so we can use reduction later
	__shared__ short smax_val[NUM_ANGLE];
	__shared__ short smax_n[NUM_ANGLE];

	// Calculate rho
	int r = round(j * hough_cos(n) + i * hough_sin(n)) + ((numrho - 1) / 2);
	
	// Update accumulator
	int val = ++adata[r + (n * numrho)];

	// Initialize shared memory with this thread's value and index
	smax_val[n] = val;
	smax_n[n] = n;

	// Wait for all threads in the block (for this point)
	// to complete calculating its rho
	__syncthreads();

	// Compare all values for this point (block) in the shared memory
	// using reduction
	for (int s = 1; s < NUM_ANGLE; s *= 2)
	{
		int index = (2 * s) * n; // Next
		if (index < NUM_ANGLE)
		{
			if (smax_val[index + s] > smax_val[index])
			{
				// Copy max value and its index
				smax_val[index] = smax_val[index + s];
				smax_n[index] = smax_n[index + s];
			}
		}

		__syncthreads();
	}

	// If we are the first thread in the block
	// write the maximum value for this point back as output
	// This diverges but its limited to just 2 writes.
	if (n == 0)
	{
		max_val[point] = smax_val[0];
		max_n[point] = smax_n[0];
	}
}

// Intial kernel updates accumulator for given point
__global__ void GPU_UpdateAccumulator(int i, int j, int numrho, short* adata, int* max_val, int* max_n)
{
	// update accumulator, find the most probable line
	//for (int n = 0; n < NUM_ANGLE; n++, adata += numrho)
	int n = threadIdx.x;
	if (n >= NUM_ANGLE)
	{
		return;
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

// Fianl kernel GPU Helper Function, calls the gpu kernel with the appropriate arguments
void UpdateAccumulatorAll(int count, int *queueXY, int numrho, short* adata, short* max_val, short* max_n)
{
	int* dev_queuexy;
	short* dev_adata;
	short* dev_max_val;
	short* dev_max_n;

	cudaEvent_t kernelStart, kernelStop, totalStart, totalStop;
	cudaEventCreate(&kernelStart);
	cudaEventCreate(&kernelStop);
	cudaEventCreate(&totalStart);
	cudaEventCreate(&totalStop);

	//start the timer here for the total time
	cudaEventRecord(totalStart, 0);

	cudaMalloc((void**)&dev_adata, NUM_ANGLE * numrho * sizeof(short));
	cudaMalloc((void**)&dev_queuexy, count * sizeof(int));
	cudaMalloc((void**)&dev_max_val, count * sizeof(short));
	cudaMalloc((void**)&dev_max_n, count * sizeof(short));

	// Copy input vectors from host memory to GPU buffers.
	cudaMemcpy(dev_adata, adata, NUM_ANGLE * numrho * sizeof(short), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_queuexy, queueXY, count * sizeof(int), cudaMemcpyHostToDevice);

	// Start the timer here for the kernel time
	cudaEventRecord(kernelStart, 0);

	GPU_UpdateAccumulatorAll << < count, NUM_ANGLE >> > (count, dev_queuexy, numrho, dev_adata, dev_max_val, dev_max_n);
	cudaDeviceSynchronize();

	// Stop kernel timer here
	cudaEventRecord(kernelStop, 0);
	cudaEventSynchronize(kernelStop);

	cudaMemcpy(adata, dev_adata, NUM_ANGLE * numrho * sizeof(short), cudaMemcpyDeviceToHost);
	cudaMemcpy(max_val, dev_max_val, count * sizeof(short), cudaMemcpyDeviceToHost);
	cudaMemcpy(max_n, dev_max_n, count * sizeof(short), cudaMemcpyDeviceToHost);

	// Stop total time here
	cudaEventRecord(totalStop, 0);
	cudaEventSynchronize(totalStop);

	// Calculate elapsed times
	float kernelTime, totalTime;
	cudaEventElapsedTime(&kernelTime, kernelStart, kernelStop);
	cudaEventElapsedTime(&totalTime, totalStart, totalStop);
	if (DEBUG)
	{
		std::cout << "Kernel Time = " << kernelTime << "ms" << std::endl;
		std::cout << "Total Time = " << totalTime << "ms" << std::endl;
	}

	cudaFree(dev_queuexy);
	cudaFree(dev_adata);
	cudaFree(dev_max_val);
	cudaFree(dev_max_n);
	cudaEventDestroy(kernelStart);
	cudaEventDestroy(totalStart);
	cudaEventDestroy(kernelStop);
	cudaEventDestroy(totalStop);
}

// Intial kernel GPU Helper Function, calls the gpu kernel with the appropriate arguments
void UpdateAccumulator(int i, int j, int numrho, short* dev_adata, int* dev_max_val, int* dev_max_n,
	short* adata, int* max_val, int* max_n, cudaEvent_t cuEvent, cudaStream_t stream1, cudaStream_t stream2)
{
	int host_max_val[1];
	int host_max_n[1];

	// Async mem copy
	cudaMemcpyAsync(dev_adata, adata, NUM_ANGLE * numrho * sizeof(short), cudaMemcpyHostToDevice, stream1);

	// sync point
	cudaEventRecord(cuEvent, stream1); // record event
	cudaStreamWaitEvent(stream2, cuEvent, 0); // wait for event in stream1

	GPU_UpdateAccumulator << < 1, NUM_ANGLE, 1, stream2 >> > (i, j, numrho, dev_adata, dev_max_val, dev_max_n);

	// Async mem copy
	cudaMemcpyAsync(adata, dev_adata, NUM_ANGLE * numrho * sizeof(short), cudaMemcpyDeviceToHost, stream1);
	cudaMemcpyAsync(host_max_val, dev_max_val, 1 * sizeof(int), cudaMemcpyDeviceToHost, stream1);
	cudaMemcpyAsync(host_max_n, dev_max_n, 1 * sizeof(int), cudaMemcpyDeviceToHost, stream1);

	*max_val = host_max_val[0];
	*max_n = host_max_n[0];
}

// Probabilistic hough line detection, optimized with CUDA
void HoughLines_GPU_v2Fast(cv::Mat& image, std::vector<cv::Vec4i>& lines)
{
	cv::Point pt;
	cv::RNG rng((uint64)-1);

	CV_Assert(image.type() == CV_8UC1);

	int width = image.cols;
	int height = image.rows;

	int numrho = cvRound(((width + height) * 2 + 1) / RHO);

	cv::Mat accum = cv::Mat::zeros(NUM_ANGLE, numrho, CV_16SC1);
	cv::Mat mask(height, width, CV_8UC1);

	uchar* mdata0 = mask.ptr();
	std::vector<int> nzlocXY;

	// stage 1. collect non-zero image points
	for (pt.y = 0; pt.y < height; pt.y++)
	{
		const uchar* data = image.ptr(pt.y);
		uchar* mdata = mask.ptr(pt.y);
		for (pt.x = 0; pt.x < width; pt.x++)
		{
			if (data[pt.x])
			{
				mdata[pt.x] = (uchar)1;
				nzlocXY.push_back(pt.x << 16 | pt.y);
			}
			else
			{
				mdata[pt.x] = 0;
			}
		}
	}

	int count = (int)nzlocXY.size();
	if (count == 0)
	{
		return;
	}

	short* adata = accum.ptr<short>();

	// stage 2. accumulator area
	short* maxNs = new short[count];
	short* maxVals = new short[count];
	UpdateAccumulatorAll(count, &nzlocXY[0], numrho, adata, maxVals, maxNs);

	// stage 2. process all the points in random order
	for (; count > 0; count--)
	{
		// choose random point out of the remaining ones
		int idx = rng.uniform(0, count);

		cv::Point point = cv::Point(nzlocXY[idx] >> 16, nzlocXY[idx] & 0xFFFF);
		cv::Point line_end[2];

		int i = point.y, j = point.x;
		int k, dx0, dy0;
		bool xflag, good_line;

		// "remove" it by overriding it with the last element
		nzlocXY[idx] = nzlocXY[count - 1];

		// check if it has been excluded already (i.e. belongs to some other line)
		if (!mdata0[i*width + j])
			continue;

		int max_n = maxNs[idx];
		if (maxVals[idx] < LINE_THRESH)
			continue;

		// from the current point walk in each direction
		// along the found line and extract the line segment
		float a = -sin(max_n * M_THETA) * IRHO;
		float b = cos(max_n * M_THETA) * IRHO;
		int x0 = j;
		int y0 = i;
		if (fabs(a) > fabs(b))
		{
			xflag = true;
			dx0 = a > 0 ? 1 : -1;
			dy0 = cvRound(b*(1 << SHIFT) / fabs(a));
			y0 = (y0 << SHIFT) + (1 << (SHIFT - 1));
		}
		else
		{
			xflag = false;
			dy0 = b > 0 ? 1 : -1;
			dx0 = cvRound(a*(1 << SHIFT) / fabs(b));
			x0 = (x0 << SHIFT) + (1 << (SHIFT - 1));
		}

		for (k = 0; k < 2; k++)
		{
			int gap = 0, x = x0, y = y0, dx = dx0, dy = dy0;

			if (k > 0)
				dx = -dx, dy = -dy;

			// walk along the line using fixed-point arithmetic,
			// stop at the image border or in case of too big gap
			for (;; x += dx, y += dy)
			{
				int i1, j1;

				if (xflag)
				{
					j1 = x;
					i1 = y >> SHIFT;
				}
				else
				{
					j1 = x >> SHIFT;
					i1 = y;
				}

				if (j1 < 0 || j1 >= width || i1 < 0 || i1 >= height)
					break;

				uchar* mdata = mdata0 + i1 * width + j1;

				// for each non-zero point:
				//    update line end,
				//    clear the mask element
				//    reset the gap
				if (*mdata)
				{
					gap = 0;
					line_end[k].y = i1;
					line_end[k].x = j1;
				}
				else if (++gap > LINE_GAP)
					break;
			}
		}

		good_line = std::abs(line_end[1].x - line_end[0].x) >= LINE_LENGTH ||
			std::abs(line_end[1].y - line_end[0].y) >= LINE_LENGTH;

		for (k = 0; k < 2; k++)
		{
			int x = x0, y = y0, dx = dx0, dy = dy0;
			if (k > 0)
			{
				dx = -dx, dy = -dy;
			}

			// walk along the line using fixed-point arithmetic,
			// stop at the image border or in case of too big gap
			for (;; x += dx, y += dy)
			{

				int i1, j1;

				if (xflag)
				{
					j1 = x;
					i1 = y >> SHIFT;
				}
				else
				{
					j1 = x >> SHIFT;
					i1 = y;
				}

				uchar* mdata = mdata0 + i1 * width + j1;

				// for each non-zero point:
				//    update line end,
				//    clear the mask element
				//    reset the gap
				if (*mdata)
				{
					if (good_line)
					{
						for (int n = 0; n < NUM_ANGLE; n++)
						{
							int r = cvRound(j1 * hough_cos(n) + i1 * hough_sin(n));
							r += (numrho - 1) / 2;
							adata[r + (n * numrho)]--;
						}
					}
					*mdata = 0;
				}

				if (i1 == line_end[k].y && j1 == line_end[k].x)
					break;
			}

			if (good_line)
			{
				cv::Vec4i lr(line_end[0].x, line_end[0].y, line_end[1].x, line_end[1].y);
				lines.push_back(lr);
				if ((int)lines.size() >= LINE_MAX)
				{
					return;
				}
			}
		}
	}
}

// Probabilistic hough line detection, 
void HoughLines_GPU_v1Slow(cv::Mat& image, std::vector<cv::Vec4i>& lines)
{
	cv::Point pt;
	cv::RNG rng((uint64)-1);

	CV_Assert(image.type() == CV_8UC1);

	int width = image.cols;
	int height = image.rows;

	int numrho = cvRound(((width + height) * 2 + 1) / RHO);

	cv::Mat accum = cv::Mat::zeros(NUM_ANGLE, numrho, CV_16SC1);
	cv::Mat mask(height, width, CV_8UC1);

	uchar* mdata0 = mask.ptr();
	std::vector<cv::Point> nzloc;

	// stage 1. collect non-zero image points
	for (pt.y = 0; pt.y < height; pt.y++)
	{
		const uchar* data = image.ptr(pt.y);
		uchar* mdata = mask.ptr(pt.y);
		for (pt.x = 0; pt.x < width; pt.x++)
		{
			if (data[pt.x])
			{
				mdata[pt.x] = (uchar)1;
				nzloc.push_back(pt);
			}
			else
			{
				mdata[pt.x] = 0;
			}
		}
	}

	int count = (int)nzloc.size();
	short* adata = accum.ptr<short>();

	short* dev_adata;
	int* dev_max_val;
	int* dev_max_n;

	// create streams, one for 
	cudaStream_t stream1, stream2;
	cudaStreamCreate(&stream1);
	cudaStreamCreate(&stream2);

	// create event; used for sync
	cudaEvent_t cuEvent;
	cudaEventCreate(&cuEvent);

	// allocate memory on device
	cudaMalloc((void**)&dev_adata, NUM_ANGLE * numrho * sizeof(short));
	cudaMalloc((void**)&dev_max_val, 1 * sizeof(int));
	cudaMalloc((void**)&dev_max_n, 1 * sizeof(int));

	// pin memory on host side
	cudaMallocHost((void**)&dev_adata, NUM_ANGLE * numrho * sizeof(short));
	cudaMallocHost((void**)&dev_max_val, 1 * sizeof(int));
	cudaMallocHost((void**)&dev_max_n, 1 * sizeof(int));

	// stage 2. process all the points in random order
	for (; count > 0; count--)
	{
		// choose random point out of the remaining ones
		int idx = rng.uniform(0, count);

		cv::Point point = nzloc[idx];
		cv::Point line_end[2];

		int i = point.y, j = point.x;
		int k, dx0, dy0;
		bool xflag, good_line;

		// "remove" it by overriding it with the last element
		nzloc[idx] = nzloc[count - 1];

		// check if it has been excluded already (i.e. belongs to some other line)
		if (!mdata0[i * width + j])
			continue;

		int max_n = 0;
		int max_val = LINE_THRESH - 1;

		// update accumulator, find the most probable line

		// ---- GPU -----
		int loc_max = 0;
		int loc_maxn = 0;

		UpdateAccumulator(i, j, numrho, dev_adata, dev_max_val, dev_max_n,
			adata, &loc_max, &loc_maxn, cuEvent, stream1, stream2);

		if (loc_max > max_val)
		{
			max_val = loc_max;
			max_n = loc_maxn;
		}

		// if it is too "weak" candidate, continue with another point
		if (max_val < LINE_THRESH)
			continue;

		// from the current point walk in each direction
		// along the found line and extract the line segment
		float a = -sin(max_n * M_THETA) * IRHO;
		float b = cos(max_n * M_THETA) * IRHO;
		int x0 = j;
		int y0 = i;
		if (fabs(a) > fabs(b))
		{
			xflag = true;
			dx0 = a > 0 ? 1 : -1;
			dy0 = cvRound(b * (1 << SHIFT) / fabs(a));
			y0 = (y0 << SHIFT) + (1 << (SHIFT - 1));
		}
		else
		{
			xflag = false;
			dy0 = b > 0 ? 1 : -1;
			dx0 = cvRound(a * (1 << SHIFT) / fabs(b));
			x0 = (x0 << SHIFT) + (1 << (SHIFT - 1));
		}

		for (k = 0; k < 2; k++)
		{
			int gap = 0, x = x0, y = y0, dx = dx0, dy = dy0;

			if (k > 0)
				dx = -dx, dy = -dy;

			// walk along the line using fixed-point arithmetic,
			// stop at the image border or in case of too big gap
			for (;; x += dx, y += dy)
			{
				uchar* mdata;
				int i1, j1;

				if (xflag)
				{
					j1 = x;
					i1 = y >> SHIFT;
				}
				else
				{
					j1 = x >> SHIFT;
					i1 = y;
				}

				if (j1 < 0 || j1 >= width || i1 < 0 || i1 >= height)
					break;

				mdata = mdata0 + i1 * width + j1;

				// for each non-zero point:
				//    update line end,
				//    clear the mask element
				//    reset the gap
				if (*mdata)
				{
					gap = 0;
					line_end[k].y = i1;
					line_end[k].x = j1;
				}
				else if (++gap > LINE_GAP)
					break;
			}
		}

		good_line = std::abs(line_end[1].x - line_end[0].x) >= LINE_LENGTH ||
			std::abs(line_end[1].y - line_end[0].y) >= LINE_LENGTH;

		for (k = 0; k < 2; k++)
		{
			int x = x0, y = y0, dx = dx0, dy = dy0;
			if (k > 0)
			{
				dx = -dx, dy = -dy;
			}

			// walk along the line using fixed-point arithmetic,
			// stop at the image border or in case of too big gap
			for (;; x += dx, y += dy)
			{
				uchar* mdata;
				int i1, j1;

				if (xflag)
				{
					j1 = x;
					i1 = y >> SHIFT;
				}
				else
				{
					j1 = x >> SHIFT;
					i1 = y;
				}

				mdata = mdata0 + i1 * width + j1;

				// for each non-zero point:
				//    update line end,
				//    clear the mask element
				//    reset the gap
				if (*mdata)
				{
					if (good_line)
					{
						for (int n = 0; n < NUM_ANGLE; n++)
						{
							int r = cvRound(j1 * hough_cos(n) + i1 * hough_sin(n));
							r += (numrho - 1) / 2;
							adata[r + (n * numrho)]--;
						}
					}
					*mdata = 0;
				}

				if (i1 == line_end[k].y && j1 == line_end[k].x)
					break;
			}

			if (good_line)
			{
				cv::Vec4i lr(line_end[0].x, line_end[0].y, line_end[1].x, line_end[1].y);
				lines.push_back(lr);
				if ((int)lines.size() >= LINE_MAX)
					return;
			}
		}
	}

	cudaFree(dev_adata);
	cudaFree(dev_max_val);
	cudaFree(dev_max_n);

	// destroy streams
	cudaStreamDestroy(stream1);
	cudaStreamDestroy(stream2);

	// destory cuda Event
	cudaEventDestroy(cuEvent);
}

// Probabilistic hough line detection, runs on CPU
void HoughLines_CPU(cv::Mat& image, std::vector<cv::Vec4i>& lines)
{
	cv::Point pt;
	cv::RNG rng((uint64)-1);

	CV_Assert(image.type() == CV_8UC1);

	int width = image.cols;
	int height = image.rows;

	int numrho = cvRound(((width + height) * 2 + 1) / RHO);

	cv::Mat accum = cv::Mat::zeros(NUM_ANGLE, numrho, CV_16SC1);
	cv::Mat mask(height, width, CV_8UC1);

	uchar* mdata0 = mask.ptr();
	std::vector<int> nzlocX, nzlocY;

	// stage 1. collect non-zero image points
	for (pt.y = 0; pt.y < height; pt.y++)
	{
		const uchar* data = image.ptr(pt.y);
		uchar* mdata = mask.ptr(pt.y);
		for (pt.x = 0; pt.x < width; pt.x++)
		{
			if (data[pt.x])
			{
				mdata[pt.x] = (uchar)1;
				nzlocX.push_back(pt.x);
				nzlocY.push_back(pt.y);
			}
			else
			{
				mdata[pt.x] = 0;
			}
		}
	}

	int count = (int)nzlocX.size();
	short* adata = accum.ptr<short>();

	// stage 2. process all the points in random order
	for (; count > 0; count--)
	{
		// choose random point out of the remaining ones
		int idx = rng.uniform(0, count);

		cv::Point point = cv::Point(nzlocX[idx], nzlocY[idx]);
		cv::Point line_end[2];

		int i = point.y, j = point.x;
		int k, dx0, dy0;
		bool xflag, good_line;

		// "remove" it by overriding it with the last element
		nzlocX[idx] = nzlocX[count - 1];
		nzlocY[idx] = nzlocY[count - 1];

		// check if it has been excluded already (i.e. belongs to some other line)
		if (!mdata0[i*width + j])
			continue;

		int max_n = 0;
		int max_val = LINE_THRESH - 1;

		// update accumulator, find the most probable line
		for (int n = 0; n < NUM_ANGLE; n++)
		{
			int r = cvRound(j * hough_cos(n) + i * hough_sin(n));
			r += (numrho - 1) / 2;
			int val = ++adata[r + (n * numrho)];
			if (val > max_val)
			{
				max_val = val;
				max_n = n;
			}
		}

		if (max_val < LINE_THRESH)
			continue;

		// from the current point walk in each direction
		// along the found line and extract the line segment
		float a = -sin(max_n * M_THETA) * IRHO;
		float b = cos(max_n * M_THETA) * IRHO;
		int x0 = j;
		int y0 = i;
		if (fabs(a) > fabs(b))
		{
			xflag = true;
			dx0 = a > 0 ? 1 : -1;
			dy0 = cvRound(b*(1 << SHIFT) / fabs(a));
			y0 = (y0 << SHIFT) + (1 << (SHIFT - 1));
		}
		else
		{
			xflag = false;
			dy0 = b > 0 ? 1 : -1;
			dx0 = cvRound(a*(1 << SHIFT) / fabs(b));
			x0 = (x0 << SHIFT) + (1 << (SHIFT - 1));
		}

		for (k = 0; k < 2; k++)
		{
			int gap = 0, x = x0, y = y0, dx = dx0, dy = dy0;

			if (k > 0)
				dx = -dx, dy = -dy;

			// walk along the line using fixed-point arithmetic,
			// stop at the image border or in case of too big gap
			for (;; x += dx, y += dy)
			{
				int i1, j1;

				if (xflag)
				{
					j1 = x;
					i1 = y >> SHIFT;
				}
				else
				{
					j1 = x >> SHIFT;
					i1 = y;
				}

				if (j1 < 0 || j1 >= width || i1 < 0 || i1 >= height)
					break;

				uchar* mdata = mdata0 + i1 * width + j1;

				// for each non-zero point:
				//    update line end,
				//    clear the mask element
				//    reset the gap
				if (*mdata)
				{
					gap = 0;
					line_end[k].y = i1;
					line_end[k].x = j1;
				}
				else if (++gap > LINE_GAP)
					break;
			}
		}

		good_line = std::abs(line_end[1].x - line_end[0].x) >= LINE_LENGTH ||
			std::abs(line_end[1].y - line_end[0].y) >= LINE_LENGTH;

		for (k = 0; k < 2; k++)
		{
			int x = x0, y = y0, dx = dx0, dy = dy0;
			if (k > 0)
			{
				dx = -dx, dy = -dy;
			}

			// walk along the line using fixed-point arithmetic,
			// stop at the image border or in case of too big gap
			for (;; x += dx, y += dy)
			{

				int i1, j1;

				if (xflag)
				{
					j1 = x;
					i1 = y >> SHIFT;
				}
				else
				{
					j1 = x >> SHIFT;
					i1 = y;
				}

				uchar* mdata = mdata0 + i1 * width + j1;

				// for each non-zero point:
				//    update line end,
				//    clear the mask element
				//    reset the gap
				if (*mdata)
				{
					if (good_line)
					{
						for (int n = 0; n < NUM_ANGLE; n++)
						{
							int r = cvRound(j1 * hough_cos(n) + i1 * hough_sin(n));
							r += (numrho - 1) / 2;
							adata[r + (n * numrho)]--;
						}
					}
					*mdata = 0;
				}

				if (i1 == line_end[k].y && j1 == line_end[k].x)
					break;
			}

			if (good_line)
			{
				cv::Vec4i lr(line_end[0].x, line_end[0].y, line_end[1].x, line_end[1].y);
				lines.push_back(lr);
				if ((int)lines.size() >= LINE_MAX)
				{
					return;
				}
			}
		}
	}
}

// Filters an image based upon a region of interest
cv::Mat filter_roi(cv::Mat &input, cv::Point top_left, cv::Point top_right, cv::Point bot_left, cv::Point bot_right) {
	cv::Point corners[1][4];
	corners[0][0] = bot_left;
	corners[0][1] = top_left;
	corners[0][2] = top_right;
	corners[0][3] = bot_right;

	const cv::Point* corner_list[1] = { corners[0] };
	int num_points = 4;
	int num_polygons = 1;

	cv::Mat mask = cv::Mat::zeros(cv::Size(input.cols, input.rows), input.type());

	cv::fillPoly(mask, corner_list, &num_points, num_polygons, cv::Scalar(255, 255, 255));
	cv::Mat output(cv::Size(mask.cols, mask.rows), input.type());

	// perform an AND operation to remove everything that isn't within the polygon
	cv::bitwise_and(input, mask, output);
	return output;
}

// Preprocesses an image
cv::Mat DoPreprocessing(const cv::Mat &input)
{
	const int ddepth = CV_16S;
	if (input.empty())
	{
		return input;
	}

	// Remove noise by blurring with a Gaussian filter
	cv::Mat srcBlurred;
	GaussianBlur(input, srcBlurred, cv::Size(KSIZE, KSIZE), 2, 2, cv::BORDER_DEFAULT);

	// Convert the image to grayscale
	cv::Mat srcGray;
	cvtColor(srcBlurred, srcGray, cv::COLOR_BGR2GRAY);

	// Run sobel edge detection
	cv::Mat grad_x, grad_y;
	cv::Sobel(srcGray, grad_x, ddepth, 1, 0, KSIZE, cv::BORDER_DEFAULT);
	cv::Sobel(srcGray, grad_y, ddepth, 0, 1, KSIZE, cv::BORDER_DEFAULT);

	// Run canny edge detection
	cv::Mat canny;
	cv::Canny(grad_x, grad_y, canny, 100, 150);
	return canny;
}

// Saves the accumulator matrix
void SaveAccumulatorMatrix(const cv::String &filename)
{
	cv::Mat image = DoPreprocessing(cv::imread(filename));
	if (image.empty())
	{
		return;
	}

	cv::Point pt;
	CV_Assert(image.type() == CV_8UC1);

	int width = image.cols;
	int height = image.rows;

	int numrho = cvRound(((width + height) * 2 + 1) / RHO);

	cv::Mat accum = cv::Mat::zeros(NUM_ANGLE, numrho, CV_16SC1);
	std::vector<int> nzlocXY;

	// stage 1. collect non-zero image points
	for (pt.y = 0; pt.y < height; pt.y++)
	{
		const uchar* data = image.ptr(pt.y);
		for (pt.x = 0; pt.x < width; pt.x++)
		{
			if (data[pt.x])
			{
				nzlocXY.push_back(pt.x << 16 | pt.y);
			}
		}
	}

	int count = (int)nzlocXY.size();
	if (count == 0)
	{
		return;
	}

	short* adata = accum.ptr<short>();

	// stage 2. accumulator area
	short* maxNs = new short[count];
	short* maxVals = new short[count];
	UpdateAccumulatorAll(count, &nzlocXY[0], numrho, adata, maxVals, maxNs);
	cv::imwrite("accumulator.png", accum);
}

// Runs a side-by-side comparison of the GPU and CPU implementation of the hough lines
void DoComparison(const cv::Mat &srcImage0)
{
	cv::Mat srcImage1 = srcImage0.clone();

	cv::Mat preprocessed0 = DoPreprocessing(srcImage0);
	cv::Mat preprocessed1 = preprocessed0.clone();

	// Run GPU probabilistic hough line detection
	auto gpuStart = std::chrono::high_resolution_clock::now();

	std::vector<cv::Vec4i> gpu_lines;
	HoughLines_GPU_v2Fast(preprocessed0, gpu_lines);

	auto gpuEnd = std::chrono::high_resolution_clock::now();
	std::chrono::duration<double, std::ratio<1, 1000>> gpu_time = gpuEnd - gpuStart;
	std::cout << "(GPU) Image Execution time: " << gpu_time.count() << "ms" << std::endl;

	// Draw lines detected 
	for (int k = 0; k < gpu_lines.size(); k++)
	{
		// this filters out border lines, probably not the best approach
		// remove the if in the event of a better filter
		cv::line(srcImage0, cv::Point(gpu_lines[k][0], gpu_lines[k][1]), cv::Point(gpu_lines[k][2],
			gpu_lines[k][3]), cv::Scalar(0, 0, 255), 3, 8);
	}

	// Output image
	cv::imwrite("gpu_detected.png", srcImage0);

	// Run CPU probabilistic hough line detection
	auto cpuStart = std::chrono::high_resolution_clock::now();

	std::vector<cv::Vec4i> cpu_lines;
	HoughLines_CPU(preprocessed1, cpu_lines);

	auto cpuEnd = std::chrono::high_resolution_clock::now();
	std::chrono::duration<double, std::ratio<1, 1000>> cpu_time = cpuEnd - cpuStart;
	std::cout << "(CPU) Image Execution time: " << cpu_time.count() << "ms" << std::endl;

	// Draw lines detected 
	for (int k = 0; k < cpu_lines.size(); k++)
	{
		// this filters out border lines, probably not the best approach
		// remove the if in the event of a better filter
		cv::line(srcImage1, cv::Point(cpu_lines[k][0], cpu_lines[k][1]), cv::Point(cpu_lines[k][2],
			cpu_lines[k][3]), cv::Scalar(0, 0, 255), 3, 8);
	}

	// Output image
	cv::imwrite("cpu_detected.png", srcImage1);
}

// Runs a side-by-side comparison of the GPU and CPU implementation of the hough lines
// Given a filename for the source image
void DoComparison(const std::string &filename)
{
	// read image
	cv::Mat srcImage = cv::imread(filename); // image
	if (srcImage.empty())
	{
		std::cout << "Cannot open the image file" << std::endl;
		return;
	}

	// Run comparison
	DoComparison(srcImage);
}

// Processes and saves the detected lines for an image
void SaveHoughLines(const cv::Mat &frameImage, const cv::Mat &roiImage, const cv::String &filename)
{
	int h = frameImage.cols;
	
	cv::Mat preprocessed = DoPreprocessing(roiImage);

	std::vector<cv::Vec4i> gpu_lines;
	HoughLines_GPU_v2Fast(preprocessed, gpu_lines);

	// remove these variables if better filtering is applied
	// used to filter out boundry lines
	const int outerboundL = BOUND_OFFSET;
	const int outerboundR = frameImage.size().width - BOUND_OFFSET;
	const int outerboundT = (T_BOUND_MOD * frameImage.size().height) + BOUND_OFFSET;
	const int outerboundB = frameImage.size().height - BOUND_OFFSET;

	// Draw lines detected, ignores border lines 
	for (int k = 0; k < gpu_lines.size(); k++)
	{
		// this filters out border lines, probably not the best approach
		// remove the if in the event of a better filter
		if (
			!(((outerboundT > gpu_lines[k][1]
				|| gpu_lines[k][1] > outerboundB) &&
				(outerboundT > gpu_lines[k][3]
					|| gpu_lines[k][3] > outerboundB)) ||
					((outerboundL > gpu_lines[k][0]
						|| gpu_lines[k][0] > outerboundR) &&
						(outerboundL > gpu_lines[k][2]
							|| gpu_lines[k][2] > outerboundR)))
			)
			cv::line(frameImage, cv::Point(gpu_lines[k][0], gpu_lines[k][1]),
				cv::Point(gpu_lines[k][2], gpu_lines[k][3]), cv::Scalar(0, 0, 255), 3, 8);
	}

	// Output image
	cv::imwrite(filename, frameImage);
}

// Processes a video and saves all processed frames
void ProcessVideo(const std::string &filename)
{
	// video test
	cv::VideoCapture cap("highway.mp4"); // video
	if (!cap.isOpened())
	{
		std::cout << "Cannot open the video file" << std::endl;
		return;
	}

	double time = 0;
	int f = 0;

	// Loop through all frames
	std::cout << "Press q to stop..." << std::endl;
	char key = ' ';
	while (key != 'q')
	{
		// Check for key
		if (kbhit())
		{
			key = getch();
		}

		cv::Mat frame;
		frame.convertTo(frame, CV_64F);

		bool bSuccess = cap.read(frame); // read a new frame from video
		if (!bSuccess)
		{
			std::cout << "Cannot read the frame from video file" << std::endl;
			break;
		}

		cv::Mat roi = filter_roi(frame, cv::Point(2, (T_BOUND_MOD * frame.size().height) + 2),
			cv::Point(frame.size().width - 2, (T_BOUND_MOD * frame.size().height) + 2),
			cv::Point(2, frame.size().height - 2),
			cv::Point(frame.size().width - 2, frame.size().height - 2));

		// do time calculations
		auto gpuStart = std::chrono::high_resolution_clock::now();

		SaveHoughLines(frame, roi, "./Frames/frame" + std::to_string(f) + ".png");

		auto gpuEnd = std::chrono::high_resolution_clock::now();
		std::chrono::duration<double, std::ratio<1, 1000>> gpu_time = gpuEnd - gpuStart;
		time += gpu_time.count();
		f++;
	}

	if (DEBUG)
	{
		std::cout << std::endl << "(GPU) Video Execution time: " << time << "ms" << std::endl;
	}
}

// Program entry point
int main()
{
	//manually trigger creation of the context, as it initiates no other activity besides the side effect
	//of kicking off context creation and initialization.
	cudaFree(0);

	// - Process video
	//ProcessVideo("highway.mp4");

	// - Do comparison
	//DoComparison("test.png");

	// - Save accumulation matrix
	//SaveAccumulatorMatrix("hill.jpg");
}