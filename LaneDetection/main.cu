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
#include <list>

struct HoughTransformComparer
{
	HoughTransformComparer(cv::Mat& _aux, int _diag) : aux(_aux), diag(_diag) {}

	inline bool operator()(int l1, int l2) const
	{
		int x1 = l1 % diag;
		int y1 = l1 / diag;
		int x2 = l2 % diag;
		int y2 = l2 / diag;
		return aux.at<int>(y1, x1) > aux.at<int>(y2, x2) || (aux.at<int>(y1, x1) == aux.at<int>(y2, x2) && l1 < l2);
	}
	cv::Mat& aux;
	const int diag;
};

double *CreateThetas(int startDeg, int stopDeg)
{
	int count = stopDeg - startDeg;
	double* thetas = new double[count];
	for (int i = 0; i < count; i++)
	{
		thetas[i] = (startDeg + i) * (3.14159265359 / 180.0);
	}

	return thetas;
}

double *CreateLinspace(int start, int stop, int count)
{
	double step = (stop - start) / (double)count;
	double* linspace = new double[count];
	for (int i = 0; i < count; i++)
	{
		linspace[i] = start + (i * step);
	}

	return linspace;
}

void FillAccumulatorMatrix(cv::Mat img, cv::Mat &accumulator, double *thetas, int df, double threshold)
{
	int width = img.cols;
	int height = img.rows;
	int diagonalLength = (int)round(sqrt(width * width + height * height));

	int accumulatorSize = 2 * diagonalLength * df;
	for (int y = 0; y < height; y++)
	{
		for (int x = 0; x < width; x++)
		{
			uchar val = img.at<uchar>(y, x);
			if (val > 5 / 255.0)
			{
				int xx = x + 1;
				int yy = y + 1;
				for (int t = 0; t < df; t++)
				{

					int rho = diagonalLength + (int)round(xx * cos(thetas[t]) + yy * sin(thetas[t]));
					accumulator.at<int>(rho, t) += 1;
				}
			}
		}
	}
}

void FindLocalMaximums(int numrho, int numangle, int threshold, cv::Mat &accumulator, std::vector<int>& sortBuffer)
{
	for (int r = 1; r < numrho - 1; r++)
	{
		for (int n = 1; n < numangle - 1; n++)
		{
			int value = accumulator.at<int>(r, n);

			int south = accumulator.at<int>(r, n - 1);
			int north = accumulator.at<int>(r, n + 1);
			int west = accumulator.at<int>(r - 1, n);
			int east = accumulator.at<int>(r + 1, n);

			if (value > threshold &&
				value > west && value > east &&
				value > north && value >= south)
			{
				sortBuffer.push_back(r * numrho + n);
			}
		}
	}
}

int main()
{
	const int ddepth = CV_16S;
	const int ksize = 3;
	const bool detectWhiteLines = true;

	cv::Mat srcImage = cv::imread("whiteline.png");
	if (srcImage.empty())
	{
		return EXIT_FAILURE;
	}

	int width = srcImage.cols;
	int height = srcImage.rows;
	int diagonalLength = (int)round(sqrt(width * width + height * height));

	// Remove noise by blurring with a Gaussian filter ( kernel size = 3 )
	cv::Mat srcBlurred;
	GaussianBlur(srcImage, srcBlurred, cv::Size(ksize, ksize), 0, 0, cv::BORDER_DEFAULT);

	// Convert the image to grayscale
	cv::Mat srcGray;
	cvtColor(srcBlurred, srcGray, cv::COLOR_BGR2GRAY);

	// Run sobel edge detection
	// cv::Mat grad_x, grad_y;
	// Sobel(srcGray, grad_x, ddepth, 1, 0, ksize, cv::BORDER_DEFAULT);
	// Sobel(srcGray, grad_y, ddepth, 0, 1, ksize, cv::BORDER_DEFAULT);

	// stage 1. fill accumulator
	double *thetas = CreateThetas(-90, 90);
	double *rhos = CreateLinspace(-diagonalLength, diagonalLength, diagonalLength * 2);
	cv::Mat accumulator = cv::Mat::zeros(2 * diagonalLength, 180, cv::DataType<int>::type);
	FillAccumulatorMatrix(srcGray, accumulator, thetas, 180, 0.5);
	cv::imwrite("acc.png", accumulator);

	// stage 2. find local maximums
	std::vector<int> maximums = std::vector<int>();
	FindLocalMaximums(diagonalLength, 180, 5, accumulator, maximums);

	// stage 3. sort the detected lines by accumulator value
	std::sort(maximums.begin(), maximums.end(), HoughTransformComparer(accumulator, diagonalLength));

	// stage 4. store the first min(total,linesMax) lines to the output buffer


	delete[] thetas;
	delete[] rhos;
}