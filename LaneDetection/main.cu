/// <summary>
/// Taken from OpenCV implementation of Hough Transform. 
/// Reworked certain functions to work with cuda. Optimizeds using
/// the following strats: 
/// </summary>
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
#include "acc_update.cuh"

void HoughLinesProbabilistic_v2(cv::Mat& image, int threshold, int lineLength, int lineGap, std::vector<cv::Vec4i>& lines, int linesMax)
{
	CV_Assert(image.type() == CV_8UC1);

	int width = image.cols;
	int height = image.rows;

	int numrho = cvRound(((width + height) * 2 + 1) / RHO);

	cv::Mat accum = cv::Mat::zeros(NUM_ANGLE, numrho, CV_32SC1);
	cv::Mat mask(height, width, CV_8UC1);

	uchar* mdata0 = mask.ptr();
	std::vector<int> nzlocX;
	std::vector<int> nzlocY;

	// stage 1. collect non-zero image points
	for (int y = 0; y < height; y++)
	{
		const uchar* data = image.ptr(y);
		uchar* mdata = mask.ptr(y);
		for (int x = 0; x < width; x++)
		{
			if (data[x])
			{
				mdata[x] = (uchar)1;
				nzlocX.push_back(x);
				nzlocY.push_back(y);
			}
			else
			{
				mdata[x] = 0;
			}
		}
	}

	int count = (int)nzlocX.size();
	int* adata = accum.ptr<int>();

	int* outX0 = new int[count];
	int* outY0 = new int[count];
	int* outX1 = new int[count];
	int* outY1 = new int[count];

	//Hough(width, height, &nzlocX[0], &nzlocY[0], count, adata, mdata0, numrho, outX0, outY0, outX1, outY1);

	delete[] outX0;
	delete[] outY0;
	delete[] outX1;
	delete[] outY1;
}

void HoughLinesProbabilistic_v1(cv::Mat& image, int threshold, int lineLength, int lineGap, std::vector<cv::Vec4i>& lines, int linesMax)
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

	//*
	// create streams, one for 
	cudaStream_t stream1, stream2;
	cudaStreamCreate(&stream1);
	cudaStreamCreate(&stream2);

	// create event; used for sync
	cudaEvent_t cuEvent;
	cudaEventCreate(&cuEvent); 
	//*/

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
		if (!mdata0[i*width + j])
			continue;

		int max_n = 0;
		int max_val = threshold - 1;

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

		// ---- CPU -----
		//for (int n = 0; n < NUM_ANGLE; n++)
		//{
		//	int r = cvRound(j * hough_cos(n) + i * hough_sin(n));
		//	r += (numrho - 1) / 2;
		//	int val = ++adata[r + (n * numrho)];
		//	if (val > max_val)
		//	{
		//		max_val = val;
		//		max_n = n;
		//	}
		//}

		// if it is too "weak" candidate, continue with another point
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
				if ((int)lines.size() >= linesMax)
					return;
			}
		}
	}

	cudaFree(dev_adata);
	cudaFree(dev_max_val);
	cudaFree(dev_max_n);

	//*
	// destroy streams
	cudaStreamDestroy(stream1);
	cudaStreamDestroy(stream2);

	// destory cuda Event
	cudaEventDestroy(cuEvent);
	//*/
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

	std::vector<cv::Vec4i> lines;
	HoughLinesProbabilistic_v1(canny, 80, 200, 10, lines, 10);

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
