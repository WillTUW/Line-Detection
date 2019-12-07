#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <opencv2/opencv.hpp>
#include <iostream>
#include <stdio.h>

void Preprocess()
{

}

void HighlightLanes()
{

}

int main()
{
	// 1. Open Image
	cv::Mat img = cv::imread("test.png");

	// 2. Preprocess

	// 3. Prepare for hough transform kernel

	// 4. Launch hough transform kernel

	// 5. Convert back into Mat for display

	// 6. Display result
	cv::namedWindow("image", cv::WINDOW_NORMAL);
	imshow("image", img);
	cv::waitKey(0);
	return 0;
}