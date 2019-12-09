
#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>
#include <stdio.h>
#include <stdarg.h>
#include <vector>

using namespace std;
using namespace cv;


string type2str(int type);

void quantize(const Mat& src, Mat& out, int quantLevel);

void accumulateMat(const Mat& lhs, const Mat& rhs, Mat& out);

void findLine(const Mat& src, Mat* dst, int m, int y, int thresh);

void getMagnitude(const Mat& src, Mat& dst);

void convertPhase(const Mat& src, Mat& out);

int main()
{
	bool debug = false;
	int ksize = 3;
	int ddepth = CV_16S;

	Mat blurredImg, src_gray;

	Mat img = imread("GetImage.jpg");
	// Check if image is loaded fine
	if (img.empty())
	{
		printf("Error opening image:\n");
		return EXIT_FAILURE;
	}

	if (debug)
	{
		imshow("image", img);
		waitKey();
	}

	// Remove noise by blurring with a Gaussian filter ( kernel size = 3 )
	GaussianBlur(img, blurredImg, Size(3, 3), 0, 0, BORDER_DEFAULT);

	// Convert the image to grayscale
	cvtColor(blurredImg, src_gray, COLOR_BGR2GRAY);
	Mat grad_x, grad_y, gmag, grad, theta, gmagQ, thetaQ;
	Mat abs_grad_x, abs_grad_y;
	Sobel(src_gray, grad_x, ddepth, 1, 0, ksize, BORDER_DEFAULT);
	Sobel(src_gray, grad_y, ddepth, 0, 1, ksize, BORDER_DEFAULT);

	if (debug) {
		imshow("Gradient X", grad_x);
		imshow("Gradient Y", grad_y);
		waitKey();
	}

	cout << grad_x.size() << ", " << grad_y.size() << endl;

	string gx = type2str(grad_x.type());
	string gy = type2str(grad_y.type());

	cout << "gx type: " << gx << endl;
	cout << "gy type: " << gy << endl;



	// convert matrices to CV_32F for magnitude and phase method
	grad_x.convertTo(grad_x, CV_32F);
	grad_y.convertTo(grad_y, CV_32F);
	//determine gradient magnitude;

	// determine theta angle in degrees and magnitude
	//requires CV_32F or CV_64F
	phase(grad_x, grad_y, theta, true);

	//convert to 1st quadrant
	convertPhase(theta, theta);
	//cout << theta << endl;

	// set magnitude using formula 
	//d = col *cos(theta) - row * sin(theta)
	gmag = Mat::zeros(theta.rows, theta.cols, CV_32F);
	getMagnitude(theta, gmag);

	//quantize gmag and theta -- gmag units of 3, theta units of 10 degrees
	gmagQ = Mat::zeros(theta.rows, theta.cols, CV_32SC1);
	thetaQ = Mat::zeros(theta.rows, theta.cols, CV_32SC1);

	quantize(gmag, gmagQ, 3);
//	cout << gmagQ << endl;
	quantize(theta, thetaQ, 10);
//	cout << thetaQ << endl;

	// find maximum value in gmag
	double minVal; double maxVal;
	Point minLoc; Point maxLoc;  Point matchLoc;

	minMaxLoc(gmagQ, &minVal, &maxVal, &minLoc, &maxLoc, Mat());
	cout << "maxVal gmagQ: " << maxVal << endl;
	int thetaBinSize = 10;
	int dBinSize = 3;
	int numCols = (90 / thetaBinSize);
	int numRows = (maxVal / dBinSize);

	Mat accumulator = Mat::zeros(numRows, numCols, CV_32SC1);
	cout << "Accumulator size: " << accumulator.size() << endl;

	accumulateMat(thetaQ, gmagQ, accumulator);

	cout << accumulator << endl;

	// find maximum value and location in the accumulator
	double minVal1; double maxVal1;
	Point minLoc1; Point maxLoc1;  Point matchLoc1;
	minMaxLoc(accumulator, &minVal1, &maxVal1, &minLoc1, &maxLoc1, Mat());
	cout << "maxVal gmagQ: " << maxVal << endl;
	cout << "M = " << maxLoc1.x * thetaBinSize << "\tY= " << maxLoc1.y * dBinSize << endl;

	return 0;
}


string type2str(int type) {
	string r;

	uchar depth = type & CV_MAT_DEPTH_MASK;
	uchar chans = 1 + (type >> CV_CN_SHIFT);

	switch (depth) {
	case CV_8U:  r = "8U"; break;
	case CV_8S:  r = "8S"; break;
	case CV_16U: r = "16U"; break;
	case CV_16S: r = "16S"; break;
	case CV_32S: r = "32S"; break;
	case CV_32F: r = "32F"; break;
	case CV_64F: r = "64F"; break;
	default:     r = "User"; break;
	}

	r += "C";
	r += (chans + '0');

	return r;
}

//this can be implemented using ParallelLoopBody https://docs.opencv.org/trunk/d7/dff/tutorial_how_to_use_OpenCV_parallel_for_.html
void quantize(const Mat& src, Mat& out, int quantLevel)
{
	cout << "Src: " << src.size() << " Out: " << out.size() << endl;
	for (int row = 0; row < src.rows; row++)
	{
		for (int col = 0; col < src.cols; col++)
		{
			int temp = abs(src.at<float>(row, col));
			if (temp > 0) {
				out.at<int>(row, col) = temp - (temp % quantLevel);
			}
		}
	}
}

void accumulateMat(const Mat& lhs, const Mat& rhs, Mat& out)
{
	cout << "LHS size: " << lhs.size() << ", RHS size: " << rhs.size() << ", out size: " << out.size() << endl;
	if (lhs.size() == rhs.size()) {
		for (int row = 0; row < lhs.rows; row++)
		{
			for (int col = 0; col < lhs.cols; col++)
			{
				int x = lhs.at<int>(row, col) % out.rows;
				int y = rhs.at<int>(row, col) % out.cols;
				//cout << "Position: " << x << ", " << y << "[" << row << ", " << col << "]" <<  endl;
				if (x > 0 && y > 0)
				{
					out.at<int>(x, y) += 1;
				}
			}
		}
	}
	else {
		cerr << "Accumulate Mat method: matrix sizes do not match." << endl;
	}

}


void findLine(const Mat& src, Mat* dst, int m, int y, int thresh)
{
	double minVal; double maxVal;
	Point minLoc; Point maxLoc;  Point matchLoc;

	minMaxLoc(src, &minVal, &maxVal, &minLoc, &maxLoc, Mat());

	vector<Point> points;
	points.push_back(maxLoc);

	int limit = m * src.cols + y;
	for (int x = 0; x < limit; x++)
	{
		//if (dst.at<float>(x * m, y) {
		//	//do something
		//}
	}

	//polyline(src, )
}


void getMagnitude(const Mat& src, Mat& dst)
{
	for (int row = 0; row < src.rows; row++)
	{
		for (int col = 0; col < src.cols; col++)
		{
			float theta = src.at<float>(row, col);
			dst.at<float>(row, col) = static_cast<float>(col * cos(theta) - row * sin(theta));
		}
	}
}
//converts to 1st quadrant angle
void convertPhase(const Mat& src, Mat& out) 
{
	for (int row = 0; row < src.rows; row++)
	{
		for (int col = 0; col < src.cols; col++)
		{
			float theta = src.at<float>(row, col);
					out.at<float>(row,col) = (fmod(theta, 90));
		}
	}
}