
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

using namespace std;
using namespace cv;


string type2str(int type);

int quantForm(float t, int quantLvl);

void quantize(const Mat& src, Mat& out, int quantLevel);

void accumulateMat(const Mat& lhs, const Mat& rhs, const Mat& mag, Mat& out, list<Point>& ptList, int thresh);

void findLine(Mat& img, Mat& accum, const Mat& gmag, list<Point>& pList, const Mat& d, const Mat& theta, const Mat& dQ, const Mat& thetaQ, int thresh);

void getDistance(const Mat& src, Mat& dst);

void convertPhase(const Mat& src, Mat& out);

void setZero(Mat& accum, int row, int col);

bool containsPt(const list<Point>& v, Point p);


int main()
{
	bool debug = false;
	int ksize = 3;
	int ddepth = CV_16S;

	Mat blurredImg, src_gray;

	Mat img = imread("GetImage.jpg", 1);
	// Check if image is loaded fine
	cout << "img: " << img.size() << endl;
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
	Mat grad_x, grad_y, gmag, grad, theta, thetaQ;
	Mat abs_grad_x, abs_grad_y;
	list<Point> ptList;
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
	magnitude(grad_x, grad_y, gmag);
	cout << "gmag size: " << gmag.size() << endl;
	//cout << gmag << endl;

	// determine theta angle in degrees and magnitude
	//requires CV_32F or CV_64F
	phase(grad_x, grad_y, theta, true);
	//cout << theta << endl;

	//convert to 1st quadrant
	convertPhase(theta, theta);
	//cout << theta << endl;

	// set distance from origin using formula 
	//d = col *cos(theta) - row * sin(theta)
	Mat d = Mat::zeros(theta.rows, theta.cols, CV_32F);



	getDistance(theta, d);

	//quantize gmag and theta -- gmag units of 3, theta units of 10 degrees
	Mat dQ = Mat::zeros(theta.rows, theta.cols, CV_32SC1);
	thetaQ = Mat::zeros(theta.rows, theta.cols, CV_32SC1);

	quantize(d, dQ, 3);
	//	cout << dQ << endl;
	quantize(theta, thetaQ, 10);
	//	cout << thetaQ << endl;

	// find maximum value in gmag
	double minVal; double maxVal;
	Point minLoc; Point maxLoc;  Point matchLoc;

	minMaxLoc(dQ, &minVal, &maxVal, &minLoc, &maxLoc, Mat());

	// find maximum value in theta
	double minValTheta; double maxValTheta;
	Point minLocTheta; Point maxLocTheta;

	minMaxLoc(thetaQ, &minValTheta, &maxValTheta, &minLocTheta, &maxLocTheta, Mat());


	cout << "maxVal dQ: " << maxVal << endl;
	cout << "maxVal theatQ: " << maxValTheta << endl;
	int thetaBinSize = 10;
	int dBinSize = 3;
	int numCols = (int)(maxValTheta / thetaBinSize);
	int numRows = (int)(maxVal / dBinSize);

	Mat accumulator = Mat::zeros(numRows, numCols, CV_32SC1);
	cout << "Accumulator size: " << accumulator.size() << endl;

	accumulateMat(thetaQ, dQ, gmag, accumulator, ptList, 1000);

	cout << accumulator << endl;

	// find maximum value and location in the accumulator
	double minVal1; double maxVal1;
	Point minLoc1; Point maxLoc1;  Point matchLoc1;
	minMaxLoc(accumulator, &minVal1, &maxVal1, &minLoc1, &maxLoc1, Mat());
	cout << "maxVal accumulator: " << maxVal1 << endl;
	cout << "maxVal Pt:" << maxLoc1 << endl;
	cout << "M = " << maxLoc1.x * thetaBinSize << "\tY= " << maxLoc1.y * dBinSize << endl;

	int dX = maxLoc1.x;
	int dY = maxLoc1.y;
	findLine(img, accumulator, gmag, ptList, d, theta, dQ, thetaQ, 20);
	//line(img, Point(dY, dX), Point(img.cols - 1, img.rows - 1), Scalar(0, 255, 0));
	// Point(img.cols - 1, (sin(maxLoc1.x * thetaBinSize*(img.cols-1))*maxLoc1.y)
	// Point(maxLoc1.y * dBinSize, maxLoc1.y * dBinSize
	//cout << "Pt List: " << ptList << endl;

	imshow("Final Output", img);
	waitKey();


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

int quantForm(float t, int quantLvl)
{
	return static_cast<int>(t - (fmod(t, quantLvl)));
}

//this can be implemented using ParallelLoopBody https://docs.opencv.org/trunk/d7/dff/tutorial_how_to_use_OpenCV_parallel_for_.html
void quantize(const Mat& src, Mat& out, int quantLevel)
{
	cout << "Src: " << src.size() << " Out: " << out.size() << endl;
	for (int row = 0; row < src.rows; row++)
	{
		for (int col = 0; col < src.cols; col++)
		{
			float temp = (abs(src.at<float>(row, col)));
			if (temp > 0) {
				out.at<int>(row, col) = quantForm(temp, quantLevel);
			}
		}
	}
}


/// adjusted row col seq here
void accumulateMat(const Mat& lhs, const Mat& rhs, const Mat& mag, Mat& out, list<Point>& ptList, int thresh)
{
	cout << "LHS size: " << lhs.size() << ", RHS size: " << rhs.size() << ", out size: " << out.size() << endl;
	if (lhs.size() == rhs.size()) {
		for (int row = 0; row < lhs.rows; row++)
		{
			for (int col = 0; col < lhs.cols; col++)
			{
				float gmag = mag.at<float>(row, col);
				if (gmag > thresh) {
					int x = lhs.at<int>(row, col) % out.rows;
					int y = rhs.at<int>(row, col) % out.cols;
					//cout << "Position: " << x << ", " << y << "[" << row << ", " << col << "]" <<  endl;
					if (x > 0 && y > 0)
					{
						out.at<int>(x, y) += 1; // static_cast<int>(gmag);
						ptList.emplace_back(Point(col, row));
					}
				}
			}
		}
	}
	else {
		cerr << "Accumulate Mat method: matrix sizes do not match." << endl;
	}

}


void findLine(Mat& img, Mat& accum, const Mat& gmag, list<Point>& pList, const Mat& d, const Mat& theta, const Mat& dQ, const Mat& thetaQ, int thresh)
{
	double minVal; double maxVal;
	Point minLoc; Point maxLoc;  Point matchLoc;
	minMaxLoc(accum, &minVal, &maxVal, &minLoc, &maxLoc, Mat());

	list<Point> tempList(pList);

	vector<Point> searchVec{
		Point(-1,-1),
		Point(0,-1),
		Point(1,-1),
		Point(-1,0),
		Point(1,0),
		Point(-1,1),
		Point(0,1),
		Point(1,1), };

	//copy(pList.begin(), pList.end(), tempList.begin());
	//sort(tempList.begin(), tempList.end());

	// while accumulator value above a certain threshold
	//while (maxVal > thresh)
	//{
	//	cout << "FindLine MaxLoc: " << maxLoc << endl;

	//	// iterate through points list
	//	for (auto& pt : tempList)
	//	{
	//		int thetaQVal = thetaQ.at<int>(pt);
	//		int dQVal = dQ.at<int>(pt);
	//		// iterate through surrounding points of pt.
	//		for (auto& sPt : searchVec)
	//		{
	//			
	//			// locate the new point
	//			Point ptPrime = pt + sPt;
	//			float dPrime, thetaPrime, gmagPrime, thetaCalc;
	//			// new point within bounds of img
	//			if (ptPrime.x > 0 && ptPrime.x < img.cols && ptPrime.y > 0 && ptPrime.y < img.rows)
	//			{
	//				// if new point is not already part of ptList
	//				if (!containsPt(tempList, ptPrime))
	//				{
	//					// gather values at new location
	//					 dPrime = d.at<float>(ptPrime);
	//					 thetaPrime = theta.at<float>(ptPrime);
	//					 gmagPrime = gmag.at<float>(ptPrime);
	//					 thetaCalc = abs(thetaPrime - thetaQVal);
	//					// determine if ptPrime meets requirements to be added 
	//					// magnitude over threshold and within 10 degrees.
	//					if ((gmagPrime > thresh) && (thetaCalc <= 10))
	//					{
	//						// find location within accumulator of bin represented
	//						int x = quantForm(thetaPrime, accum.cols)% accum.cols;
	//						int y = quantForm(dPrime, accum.rows)% accum.rows;
	//						// place ptPrime into pt list
	//						tempList.emplace_back(ptPrime);
	//						// remove value from accumulator since accounted for
	//						setZero(accum, y, x);
	//					}
	//				}
	//			}
	//		}
	//		// remove accumulator val for point already in list.
	//		int xPt = thetaQVal % accum.cols;
	//		int yPt = dQVal % accum.rows;
	//		cout << "AccumValue removed: " << accum.at<int>(yPt, xPt) << endl;
	//		setZero(accum, xPt, yPt);
	//		cout << "  postRemoval: " << accum.at<int>(yPt, xPt) << "(" << yPt << ", " << xPt << ")" << endl;;

	//		//determine new max value in accumulator
	//		minMaxLoc(accum, &minVal, &maxVal, &minLoc, &maxLoc, Mat());
	//	}

	//}
	//cout << "fin" << endl;
	// finished iterating through accumulator
	//sort(tempList.begin(), tempList.end());

	//draw line
	for (auto& x : tempList)
	{
		// new point within bounds of img
		if (x.x > 0 && x.x < img.cols && x.y > 0 && x.y < img.rows)
		{
			img.at<Vec3b>(x)[0] = 0;
			img.at<Vec3b>(x)[1] = 255;
			img.at<Vec3b>(x)[2] = 0;
		}
	}
	//vector<Point>finalPts(tempList.begin(), tempList.end());
	//polylines(img, finalPts, false, Scalar(0, 255, 0));
}

bool containsPt(const list<Point>& v, Point p)
{
	if (std::find(v.begin(), v.end(), p) != v.end()) {
		return true;
	}
	else {
		return false;
	}
}


void getDistance(const Mat& src, Mat& dst)
{
	for (int row = 0; row < src.rows; row++)
	{
		for (int col = 0; col < src.cols; col++)
		{
			float theta = src.at<float>(row, col);
			dst.at<float>(row, col) = static_cast<float>(abs(col * cos(theta) - row * sin(theta)));
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
			out.at<float>(row, col) = (fmod(theta, 90));
		}
	}
}

void setZero(Mat& accum, int row, int col)
{
	accum.at<int>(col, row) = 0;
}