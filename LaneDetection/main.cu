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
#include <stdarg.h>
#include <vector>
#include <algorithm>
#include <chrono>
#include <list>

constexpr auto SHIFT = 16;
constexpr auto M_PI = 3.14159265359;
#define M_THETA (M_PI / 180)
constexpr auto RHO = 1.0;
#define IRHO (1 / RHO)
#define hough_cos(x) (cos(x * M_THETA) * IRHO)
#define hough_sin(x) (sin(x * M_THETA) * IRHO)
constexpr auto NUM_ANGLE = 180;

__global__ void GPU_UpdateAccumulatorAll(int count, int *queueXY, int numrho, short* adata, short* max_val, short* max_n)
{
	// update accumulator, find the most probable line
	const int point = blockIdx.x;
	if (point >= count)
	{
		return;
	}

	const int j = queueXY[point] >> 16;
	const int i = queueXY[point] & 0xFFFF;

	const int n = threadIdx.x;
	if (n >= NUM_ANGLE)
	{
		//wot
		return; //No no you right we want to use return to get out of this func :thumbsup:
	}

	__shared__ short smax_val[NUM_ANGLE];
	__shared__ short smax_n[NUM_ANGLE];

	int r = round(j * hough_cos(n) + i * hough_sin(n)) + ((numrho - 1) / 2);
	int val = ++adata[r + (n * numrho)];

	smax_val[n] = val;
	smax_n[n] = n;
	__syncthreads();

#pragma unroll
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
		max_val[point] = smax_val[0];
		max_n[point] = smax_n[0];
	}
}

void _UpdateAccumulatorAll(int count, int *queueXY, int numrho, short* adata, short* max_val, short* max_n)
{
	int* dev_queuexy;
	short* dev_adata;
	short* dev_max_val;
	short* dev_max_n;

	cudaMalloc((void**)&dev_adata, NUM_ANGLE * numrho * sizeof(short));
	cudaMalloc((void**)&dev_queuexy, count * sizeof(int));
	cudaMalloc((void**)&dev_max_val, count * sizeof(short));
	cudaMalloc((void**)&dev_max_n, count * sizeof(short));

	// Copy input vectors from host memory to GPU buffers.
	cudaMemcpy(dev_adata, adata, NUM_ANGLE * numrho * sizeof(short), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_queuexy, queueXY, count * sizeof(int), cudaMemcpyHostToDevice);

	GPU_UpdateAccumulatorAll <<< count, NUM_ANGLE >>> (count, dev_queuexy, numrho, dev_adata, dev_max_val, dev_max_n);

	cudaDeviceSynchronize();

	cudaMemcpy(adata, dev_adata, NUM_ANGLE * numrho * sizeof(short), cudaMemcpyDeviceToHost);
	cudaMemcpy(max_val, dev_max_val, count * sizeof(short), cudaMemcpyDeviceToHost);
	cudaMemcpy(max_n, dev_max_n, count * sizeof(short), cudaMemcpyDeviceToHost);

	cudaFree(dev_queuexy);
	cudaFree(dev_adata);
	cudaFree(dev_max_val);
	cudaFree(dev_max_n);
}

void UpdateAccumulatorAll(int count, int *queueXY, int numrho, short* adata, short* max_val, short* max_n)
{
	// create streams, one for 
	cudaStream_t stream1, stream2;
	cudaStreamCreate(&stream1);
	cudaStreamCreate(&stream2);

	// create event; used for sync
	cudaEvent_t cuEvent;
	cudaEventCreate(&cuEvent);
	//*/

	int* dev_queuexy;
	short* dev_adata;
	short* dev_max_val;
	short* dev_max_n;

	// allocate memory on device
	cudaMalloc((void**)&dev_adata, NUM_ANGLE * numrho * sizeof(short));
	cudaMalloc((void**)&dev_queuexy, count * sizeof(int));
	cudaMalloc((void**)&dev_max_val, count * sizeof(short));
	cudaMalloc((void**)&dev_max_n, count * sizeof(short));

	// pin memory on host side
	cudaMallocHost((void**)&dev_adata, NUM_ANGLE * numrho * sizeof(short));
	cudaMallocHost((void**)&dev_queuexy, count * sizeof(int));
	cudaMallocHost((void**)&dev_max_val, count * sizeof(short));
	cudaMallocHost((void**)&dev_max_n, count * sizeof(short));

	// Async mem copy
	cudaMemcpyAsync(dev_adata, adata, NUM_ANGLE * numrho * sizeof(short), cudaMemcpyHostToDevice, stream1);
	cudaMemcpyAsync(dev_queuexy, queueXY, count * sizeof(int), cudaMemcpyHostToDevice, stream1);

	// sync point
	cudaEventRecord(cuEvent, stream1); // record event
	cudaStreamWaitEvent(stream2, cuEvent, 0); // wait for event in stream1

	GPU_UpdateAccumulatorAll <<< count, NUM_ANGLE, 1, stream2 >>> (count, dev_queuexy, numrho, dev_adata, dev_max_val, dev_max_n);

	cudaMemcpyAsync(adata, dev_adata, NUM_ANGLE * numrho * sizeof(short), cudaMemcpyDeviceToHost, stream1);
	cudaMemcpyAsync(max_val, dev_max_val, count * sizeof(short), cudaMemcpyDeviceToHost, stream1);
	cudaMemcpyAsync(max_n, dev_max_n, count * sizeof(short), cudaMemcpyDeviceToHost, stream1);

	cudaFree(dev_queuexy);
	cudaFree(dev_adata);
	cudaFree(dev_max_val);
	cudaFree(dev_max_n);

	// destroy streams
	cudaStreamDestroy(stream1);
	cudaStreamDestroy(stream2);

	// destory cuda Event
	cudaEventDestroy(cuEvent);
}

void HoughLines_GPU(cv::Mat& image, int threshold, int lineLength, int lineGap, std::vector<cv::Vec4i>& lines, int linesMax)
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
	short* adata = accum.ptr<short>();

	short* dev_adata;
	short* dev_max_val;
	short* dev_max_n;

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
		if (maxVals[idx] < threshold)
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
				else if (++gap > lineGap)
					break;
			}
		}

		good_line = std::abs(line_end[1].x - line_end[0].x) >= lineLength ||
			std::abs(line_end[1].y - line_end[0].y) >= lineLength;

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
				if ((int)lines.size() >= linesMax)
					return;
			}
		}
	}
}

void HoughLines_CPU(cv::Mat& image, int threshold, int lineLength, int lineGap, std::vector<cv::Vec4i>& lines, int linesMax)
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
		int max_val = threshold - 1;

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

		if (max_val < threshold)
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
				else if (++gap > lineGap)
					break;
			}
		}

		good_line = std::abs(line_end[1].x - line_end[0].x) >= lineLength ||
			std::abs(line_end[1].y - line_end[0].y) >= lineLength;

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
				if ((int)lines.size() >= linesMax)
					return;
			}
		}
	}
}

int main()
{
	const int ddepth = CV_16S;
	const int ksize = 3;

	cv::Mat srcImage = cv::imread("test.png");
	if (srcImage.empty())
	{
		return EXIT_FAILURE;
	}

	// Remove noise by blurring with a Gaussian filter ( kernel size = 3 )
	cv::Mat srcBlurred;
	GaussianBlur(srcImage, srcBlurred, cv::Size(ksize, ksize), 0, 0, cv::BORDER_DEFAULT);

	// Convert the image to grayscale
	cv::Mat srcGray;
	cvtColor(srcBlurred, srcGray, cv::COLOR_BGR2GRAY);

	// Run sobel edge detection
	cv::Mat grad_x, grad_y;
	cv::Sobel(srcGray, grad_x, ddepth, 1, 0, ksize, cv::BORDER_DEFAULT);
	cv::Sobel(srcGray, grad_y, ddepth, 0, 1, ksize, cv::BORDER_DEFAULT);

	// Run canny edge detection
	cv::Mat canny;
	cv::Canny(grad_x, grad_y, canny, 100, 150);
	cv::imwrite("canny.png", canny);

	// Run probabilistic hough line detection
	auto start = std::chrono::high_resolution_clock::now();

	//
	std::vector<cv::Vec4i> lines;
	HoughLines_GPU(canny, 80, 200, 10, lines, 10);

	auto end = std::chrono::high_resolution_clock::now();
	std::chrono::duration<double, std::ratio<1, 1000>> time = end - start;
	std::cout << "Execution time: " << time.count() << "ms" << std::endl;

	// Draw lines detected 
	for (int k = 0; k < lines.size(); k++)
	{
		cv::line(srcImage, cv::Point(lines[k][0], lines[k][1]), cv::Point(lines[k][2],
			lines[k][3]), cv::Scalar(0, 0, 255), 3, 8);
	}

	// Output image
	cv::imwrite("detected.png", srcImage);
}