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

#define M_PI 3.14159265359

void HoughLinesProbabilistic(cv::Mat& image, float rho, float theta, int threshold, int lineLength, int lineGap,
	std::vector<cv::Vec4i>& lines, int linesMax)
{
	cv::Point pt;
	float irho = 1 / rho;
	cv::RNG rng((uint64)-1);

	CV_Assert(image.type() == CV_8UC1);

	int width = image.cols;
	int height = image.rows;

	int numangle = cvRound(CV_PI / theta);
	int numrho = cvRound(((width + height) * 2 + 1) / rho);

	cv::Mat accum = cv::Mat::zeros(numangle, numrho, CV_32SC1);
	cv::Mat mask(height, width, CV_8UC1);
	std::vector<float> trigtab(numangle * 2);

	for (int n = 0; n < numangle; n++)
	{
		trigtab[n * 2] = (float)(cos((double)n*theta) * irho);
		trigtab[n * 2 + 1] = (float)(sin((double)n*theta) * irho);
	}

	const float* ttab = &trigtab[0];
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
				mdata[pt.x] = 0;
		}
	}

	int count = (int)nzloc.size();

	// stage 2. process all the points in random order
	for (; count > 0; count--)
	{
		// choose random point out of the remaining ones
		int idx = rng.uniform(0, count);
		int max_val = threshold - 1, max_n = 0;
		cv::Point point = nzloc[idx];
		cv::Point line_end[2];
		float a, b;
		int* adata = accum.ptr<int>();
		int i = point.y, j = point.x, k, x0, y0, dx0, dy0, xflag;
		int good_line;
		const int shift = 16;

		// "remove" it by overriding it with the last element
		nzloc[idx] = nzloc[count - 1];

		// check if it has been excluded already (i.e. belongs to some other line)
		if (!mdata0[i*width + j])
			continue;

		// update accumulator, find the most probable line
		for (int n = 0; n < numangle; n++, adata += numrho)
		{
			int r = cvRound(j * ttab[n * 2] + i * ttab[n * 2 + 1]);
			r += (numrho - 1) / 2;
			int val = ++adata[r];
			if (max_val < val)
			{
				max_val = val;
				max_n = n;
			}
		}

		// if it is too "weak" candidate, continue with another point
		if (max_val < threshold)
			continue;

		// from the current point walk in each direction
		// along the found line and extract the line segment
		a = -ttab[max_n * 2 + 1];
		b = ttab[max_n * 2];
		x0 = j;
		y0 = i;
		if (fabs(a) > fabs(b))
		{
			xflag = 1;
			dx0 = a > 0 ? 1 : -1;
			dy0 = cvRound(b*(1 << shift) / fabs(a));
			y0 = (y0 << shift) + (1 << (shift - 1));
		}
		else
		{
			xflag = 0;
			dy0 = b > 0 ? 1 : -1;
			dx0 = cvRound(a*(1 << shift) / fabs(b));
			x0 = (x0 << shift) + (1 << (shift - 1));
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
					i1 = y >> shift;
				}
				else
				{
					j1 = x >> shift;
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
					i1 = y >> shift;
				}
				else
				{
					j1 = x >> shift;
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
						adata = accum.ptr<int>();
						for (int n = 0; n < numangle; n++, adata += numrho)
						{
							int r = cvRound(j1 * ttab[n * 2] + i1 * ttab[n * 2 + 1]);
							r += (numrho - 1) / 2;
							adata[r]--;
						}
					}
					*mdata = 0;
				}

				if (i1 == line_end[k].y && j1 == line_end[k].x)
					break;
			}
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

int main()
{
	const int ddepth = CV_16S;
	const int ksize = 1;
	const bool detectWhiteLines = true;

	cv::Mat srcImage = cv::imread("test.png");
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
	cv::Mat grad_x, grad_y;
	cv::Sobel(srcGray, grad_x, ddepth, 1, 0, ksize, cv::BORDER_DEFAULT);
	cv::Sobel(srcGray, grad_y, ddepth, 0, 1, ksize, cv::BORDER_DEFAULT);

	// Run canny edge detection
	cv::Mat canny;
	cv::Canny(grad_x, grad_y, canny, 100, 150);
	cv::imwrite("canny.png", canny);

	std::vector<cv::Vec4i> lines;
	HoughLinesProbabilistic(canny, 1, CV_PI / 180, 80, 200, 150, lines, 10);

	for (int k = 0; k < lines.size(); k++)
	{
		cv::line(srcImage, cv::Point(lines[k][0], lines[k][1]), cv::Point(lines[k][2], lines[k][3]), cv::Scalar(0, 0, 255), 3, 8);
	}

	cv::imwrite("detected.png", srcImage);
}