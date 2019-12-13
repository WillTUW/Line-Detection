#include "hough.cuh"

__global__ void GPU_Hough(int width, int height, int *queueX, int *queueY,
	int *adata, unsigned char* maskData, int numrho, int* outX0, int *outY0, int *outX1, int *outY1)
{
	// stage 2. process all the points in random order
	//for (int idx = 0; idx < count; idx++)
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	// choose random point out of the remaining ones
	//cv::Point point = nzloc[idx];
	int pointX = queueX[idx];
	int pointY = queueY[idx];

	//cv::Point line_end[2];
	int line_endX[2];
	int line_endY[2];

	int i = pointY, j = pointX;
	int k, dx0, dy0;
	bool xflag, is_line_long_enough;

	// "remove" it by overriding it with the last element
	//nzloc[idx] = nzloc[count - 1];

	// check if it has been excluded already (i.e. belongs to some other line)
	//if (!mdata0[i*width + j])
	//	continue;

	int max_n = 0;
	int max_val = THRESHOLD - 1;

	// update accumulator, find the most probable line
	//int loc_max = 0;
	//int loc_maxn = 0;
	//UpdateAccumulator(i, j, numrho, adata, &loc_max, &loc_maxn);
	//if (loc_max > max_val)
	//{
	//	max_val = loc_max;
	//	max_n = loc_maxn;
	//}

	for (int n = 0; n < NUM_ANGLE; n++)
	{
		int r = round(j * hough_cos(n) + i * hough_sin(n));
		r += (numrho - 1) / 2;
		int val = ++adata[r + (n * numrho)];
		if (val > max_val)
		{
			max_val = val;
			max_n = n;
		}
	}

	// if it is too "weak" candidate, continue with another point
	if (max_val < THRESHOLD)
		return;

	// from the current point walk in each direction
	// along the found line and extract the line segment
	float a = -sin(max_n * M_THETA) * IRHO;
	float b = cos(max_n * M_THETA) * IRHO;
	int x0 = j;
	int y0 = i;
	if (fabsf(a) > fabsf(b))
	{
		xflag = true;
		dx0 = a > 0 ? 1 : -1;
		dy0 = round(b*(1 << SHIFT) / fabsf(a));
		y0 = (y0 << SHIFT) + (1 << (SHIFT - 1));
	}
	else
	{
		xflag = false;
		dy0 = b > 0 ? 1 : -1;
		dx0 = round(a*(1 << SHIFT) / fabsf(b));
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

			unsigned char* mdata = maskData + i1 * width + j1;

			// for each non-zero point:
			//    update line end,
			//    clear the mask element
			//    reset the gap
			if (*mdata)
			{
				gap = 0;
				line_endY[k] = i1;
				line_endX[k] = j1;
			}
			else if (++gap > LINE_GAP)
				break;
		}
	}

	is_line_long_enough = abs(line_endX[1] - line_endX[0]) >= LINE_LENGTH ||
		abs(line_endY[1] - line_endY[0]) >= LINE_LENGTH;

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

			unsigned char* mdata = maskData + i1 * width + j1;

			// for each non-zero point:
			//    update line end,
			//    clear the mask element
			//    reset the gap
			if (*mdata)
			{
				if (is_line_long_enough)
				{
					for (int n = 0; n < NUM_ANGLE; n++)
					{
						int r = round(j1 * hough_cos(n) + i1 * hough_sin(n));
						r += (numrho - 1) / 2;
						adata[r + (n * numrho)]--;
					}
				}
				*mdata = 0;
			}

			if (i1 == line_endY[k] && j1 == line_endX[k])
				break;
		}

		if (is_line_long_enough)
		{
			//cv::Vec4i lr(line_endX[0], line_endY[0], line_endX[1], line_endY[1]);
			//lines.push_back(lr);
			//if ((int)lines.size() >= linesMax)
			//	return;
		}
	}
}

void Hough(int width, int height, int *queueX, int *queueY,
	int *adata, unsigned char* maskData, int numrho, int* outX0, int *outY0, int *outX1, int *outY1)
{
	//int* dev_queueX;
	//int* dev_queueY;
	//int* dev_adata;
	//unsigned char* dev_maskData;

	//cudaMalloc((void**)&dev_adata, NUM_ANGLE * numrho * sizeof(int));
	//cudaMalloc((void**)&dev_max_val, 1 * sizeof(int));
	//cudaMalloc((void**)&dev_max_n, 1 * sizeof(int));

	//// Copy input vectors from host memory to GPU buffers.
	//cudaMemcpy(dev_adata, adata, NUM_ANGLE * numrho * sizeof(int), cudaMemcpyHostToDevice);

	//GPU_UpdateAccumulator << < 1, NUM_ANGLE >> > (i, j, numrho, dev_adata, dev_max_val, dev_max_n);

	//cudaDeviceSynchronize();

	//cudaMemcpy(adata, dev_adata, NUM_ANGLE * numrho * sizeof(int), cudaMemcpyDeviceToHost);
	//cudaMemcpy(host_max_val, dev_max_val, 1 * sizeof(int), cudaMemcpyDeviceToHost);
	//cudaMemcpy(host_max_n, dev_max_n, 1 * sizeof(int), cudaMemcpyDeviceToHost);

	//cudaFree(dev_adata);
	//cudaFree(dev_max_val);
	//cudaFree(dev_max_n);

	//*max_val = host_max_val[0];
	//*max_n = host_max_n[0];
}