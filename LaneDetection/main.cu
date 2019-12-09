
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <opencv2/opencv.hpp>
#include <iostream>
#include <stdio.h>

int main()
{
	cv::Mat img = cv::imread("test.png");
	cv::namedWindow("image", cv::WINDOW_NORMAL);
	imshow("image", img);
	cv::waitKey(0);
	/// <summary>
	/// ////////////
	/// </summary>
	/// <returns></returns>
	return 0;
}