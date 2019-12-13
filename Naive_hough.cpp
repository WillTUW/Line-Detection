// Naive_hough.cpp
//
// Css490 - HighPerformance Computing
// Jason White, Will Thomas, Austin Green, Christian Rahmel
//
// Final Project - This proram is a naive cpu based approach to a hough transform for linear lines.  
// Our group did not end up using this as the model to improve upon.
// The accumulator in this project is modeled after O'Gorman's and Clowe's verion of the Hough Method.

#include <opencv2/opencv.hpp>
#include <time.h>

using namespace std;
using namespace cv;

// utility method to determine depth of Mat
//  ex.   string gx = type2str(grad_x.type());
string type2str(int type);

// method used to reduce resolution of accumulator.
int quantForm(float t, int quantLvl);

// method quantizes matrix used with O'Gorman and Clowes method.
// variables: src: matrix input, out: matrix output, quantLevel: number to fit to.
// formula used is t - (fmod(t, quantLvl))
void quantize(const Mat& src, Mat& out, int quantLevel);

// method accumulates lines from Hough transform into bins  y axis = rho  x axis = theta normalized to 1st quadrant (0-90degrees)
// places ptlist in <pair(imgloc), pair(accumloc)> order
void accumulateMat(const Mat& thetaQ, const Mat& rhoQ, const Mat& gmag, Mat& accumulator, list< std::pair< std::pair<int, int>, std::pair<int, int> >>& ptList, int thresh);

// for all accumulator bins above a threshold, method gathers points along this line from the pList,  sorts by size,
// sorts by size,  chooses first and last points and draws a line between these points.
// variables accum: Matrix accumulator,  dst: Matrix output image,  thold: int threshold value , pList: list<pair< int pairs>> pointList from accumulator
void findLine(const Mat& accum, Mat& dst, int tHold, list< std::pair< std::pair<int, int>, std::pair<int, int> >> pList);

// method determines distance from image origin.
// variables:  src: Mat theta Matrix,  dst: Mat rho Matrix.
void getDistance(const Mat& src, Mat& dst);

// converts to 1st quadrant angle
void convertPhase(const Mat& theta, Mat& out);

// method determins of point p is contained in list v
bool containsPt(const list<Point>& v, Point p);


int main()
{
	bool debug = false;
	int ksize = 3;
	int ddepth = CV_16S;

	Mat blurredImg, src_gray, grad_x, grad_y, gmag, grad, theta, thetaQ;
	Mat abs_grad_x, abs_grad_y;
	list< std::pair< std::pair<int, int>, std::pair<int, int> >> ptList;

	//Mat img = imread("GetImage.jpg", 1);
	Mat img = imread("highway.jpg", 1);

	// Check if image is loaded fine
	if (img.empty())
	{
		printf("Error opening image:\n");
		return EXIT_FAILURE;
	}

	if (debug)
	{
		cout << "img: " << img.size() << endl;
		cout << "img type: " << type2str(img.type()) << endl;
		imshow("image", img);
		waitKey();
	}

	//start clock
	clock_t t = clock();

	// Remove noise by blurring with a Gaussian filter ( kernel size = 3x3 )
	GaussianBlur(img, blurredImg, Size(3, 3), 2, 2, BORDER_DEFAULT);

	// Convert the image to grayscale
	cvtColor(blurredImg, src_gray, COLOR_BGR2GRAY);

	// Find gradients
	Sobel(src_gray, grad_x, ddepth, 1, 0, ksize, BORDER_DEFAULT);
	Sobel(src_gray, grad_y, ddepth, 0, 1, ksize, BORDER_DEFAULT);

	if (debug) {
		string gx = type2str(grad_x.type());
		string gy = type2str(grad_y.type());
		cout << "gx type: " << gx << endl;
		cout << "gy type: " << gy << endl;
		imshow("Gradient X", grad_x);
		imshow("Gradient Y", grad_y);
		waitKey();
	}

	// convert matrices to CV_32F for magnitude and phase method
	grad_x.convertTo(grad_x, CV_32F);
	grad_y.convertTo(grad_y, CV_32F);

	//determine gradient magnitude;
	magnitude(grad_x, grad_y, gmag);

	// determine theta angle in degrees and magnitude
	//requires CV_32F or CV_64F
	phase(grad_x, grad_y, theta, true);

	//convert to 1st quadrant
	convertPhase(theta, theta);

	// set distance from origin using formula 
	//rho = col *cos(theta*pi/180) - row * sin(theta*pi/180)
	Mat rho = Mat::zeros(theta.rows, theta.cols, CV_32F);
	getDistance(theta, rho);

	//quantize gmag and theta -- gmag units of 3, theta units of 10 degrees
	Mat rhoQ = Mat::zeros(theta.rows, theta.cols, CV_32SC1);
	thetaQ = Mat::zeros(theta.rows, theta.cols, CV_32SC1);
	int thetaBinSize = 10;
	int dBinSize = 3;
	quantize(rho, rhoQ, dBinSize);
	quantize(theta, thetaQ, thetaBinSize);

	// find maximum value in gmag
	double minVal; double maxVal;
	Point minLoc; Point maxLoc;  Point matchLoc;
	minMaxLoc(rhoQ, &minVal, &maxVal, &minLoc, &maxLoc, Mat());

	int numCols = (int)(90 / thetaBinSize);
	int numRows = (int)(maxVal / dBinSize);

	Mat accumulator = Mat::zeros(numCols, numRows, CV_32SC1);
	accumulateMat(thetaQ, rhoQ, gmag, accumulator, ptList, 1950);

	// find maximum value and location in the accumulator
	double minVal1; double maxVal1;
	Point minLoc1; Point maxLoc1;  Point matchLoc1;
	minMaxLoc(accumulator, &minVal1, &maxVal1, &minLoc1, &maxLoc1, Mat());

	if (debug)
	{
		cout << "maxVal rhoQ: " << maxVal << endl;
		cout << "Accumulator size: " << accumulator.size() << endl;
		cout << accumulator << endl;
		cout << "maxVal accumulator: " << maxVal1 << endl;
		cout << "maxVal Pt:" << maxLoc1 << endl;
	}

	// draw lines found
	findLine(accumulator, img, 0, ptList);

	// calculate time elapsed
	t = clock() - t;
	float seconds = ((float)t) / CLOCKS_PER_SEC;
	cout << " took " << seconds << " SECONDS." << endl;

	imshow("Final Output", img);
	waitKey();

	return 0;
}

// method used to reduce resolution of accumulator.
int quantForm(float t, int quantLvl)
{
	return static_cast<int>(t - (fmod(t, quantLvl)));
}

//this can be implemented using ParallelLoopBody https://docs.opencv.org/trunk/d7/dff/tutorial_how_to_use_OpenCV_parallel_for_.html
// method quantizes matrix used with O'Gorman and Clowes method.
// variables: src: matrix input, out: matrix output, quantLevel: number to fit to.
// formula used is t - (fmod(t, quantLvl))
void quantize(const Mat& src, Mat& out, int quantLevel)
{
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

// method accumulates lines from Hough transform into bins  y axis = rho  x axis = theta normalized to 1st quadrant (0-90degrees)
// places ptlist in <pair(imgloc), pair(accumloc)> order
void accumulateMat(const Mat& thetaQ, const Mat& rhoQ, const Mat& gmag, Mat& accumulator, list< std::pair< std::pair<int, int>, std::pair<int, int> >>& ptList, int thresh)
{
	if (thetaQ.size() == rhoQ.size()) {
		for (int row = 0; row < thetaQ.rows; row++)
		{
			for (int col = 0; col < thetaQ.cols; col++)
			{
				float gmagVal = gmag.at<float>(row, col);
				if (gmagVal > thresh) {
					int x = thetaQ.at<int>(row, col) % accumulator.rows;
					int y = rhoQ.at<int>(row, col) % accumulator.cols;

					if (x > 0 && y > 0)
					{
						accumulator.at<int>(x, y) += 1; // static_cast<int>(gmag);
						ptList.emplace_back(std::pair< std::pair<int, int>, std::pair<int, int> >(std::pair<int, int>(col, row), std::pair<int, int>(x, y)));
					}
				}
			}
		}
	}
	else {
		cerr << "Accumulate Mat method: matrix sizes do not match." << endl;
	}
}

// for all accumulator bins above a threshold, method gathers points along this line from the pList,  sorts by size,
// sorts by size,  chooses first and last points and draws a line between these points.
// variables accum: Matrix accumulator,  dst: Matrix output image,  thold: int threshold value , pList: list<pair< int pairs>> pointList from accumulator
void findLine(const Mat& accum, Mat& dst, int tHold, list< std::pair< std::pair<int, int>, std::pair<int, int> >> pList)
{
	float toRads = CV_PI / 180;
	for (int accRow = 0; accRow < accum.rows; accRow++)
	{
		for (int accCol = 0; accCol < accum.cols; accCol++)
		{
			int val = accum.at<int>(accRow, accCol);
			if (val > tHold)
			{
				vector<Point> pointList;
				for (auto x : pList)
				{
					std::pair<int, int> ptAcc(x.second);
					std::pair<int, int> ptImg(x.first);

					// point in list within family
					if (ptAcc.first == accRow && ptAcc.second == accCol)
					{
						pointList.emplace_back(Point(ptImg.first, ptImg.second));
					}
				}

				//cout << pointList << endl; 
				std::sort(pointList.begin(), pointList.end(), [](const auto& p1, const auto& p2) {
					if (p1.x == p2.x)
					{
						return p1.y < p2.y;
					}
					return p1.x < p2.x;
					});

				Point beg = pointList[0];
				Point end = pointList[pointList.size() - 1];

				// draw segment
				line(dst, beg, end, Scalar(0, 255, 0), 10);
			}
		}
	}
}

// method determins of point p is contained in list v
bool containsPt(const list<Point>& v, Point p)
{
	if (std::find(v.begin(), v.end(), p) != v.end()) {
		return true;
	}
	else {
		return false;
	}
}

// method determines distance from image origin.
// variables:  src: Mat theta Matrix,  dst: Mat rho Matrix.
void getDistance(const Mat& src, Mat& dst)
{
	float toRads = CV_PI / 180;
	for (int row = 0; row < src.rows; row++)
	{
		for (int col = 0; col < src.cols; col++)
		{
			float theta = src.at<float>(row, col);
			dst.at<float>(row, col) = static_cast<float>(abs(col * cos(theta * (toRads)) - row * sin(theta * toRads)));
		}
	}
}

// converts to 1st quadrant angle
void convertPhase(const Mat& theta, Mat& out)
{
	for (int row = 0; row < theta.rows; row++)
	{
		for (int col = 0; col < theta.cols; col++)
		{
			float thetaVal = theta.at<float>(row, col);
			out.at<float>(row, col) = (fmod(thetaVal, 90));
		}
	}
}

// utility method to determine depth of Mat
//  ex.   string gx = type2str(grad_x.type());
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
