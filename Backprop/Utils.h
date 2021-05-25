#pragma once

#include <vector>
#include "MNIST.h"
#include "Display.h"

MatrixXd buildMapfromData(std::vector <MatrixXd> & samples, int sample_width, int sample_height) 
{
	int n_rows = std::ceil(std::sqrt(samples.size()));
	int n_cols = (samples.size() + n_rows - 1) / n_rows;

	int map_width  = sample_width  * n_rows;
	int map_height = sample_height * n_cols;

	MatrixXd map(map_width, map_height);
	map.setZero();

	for (int n = 0; n < (int) samples.size(); n++) 
	{
		//std::cout << n << std::endl;
		int i = (n % n_cols);
		int j = (n / n_cols);

		MatrixXd sample = samples[n];

		for (int di = 0; di < sample_width; di++) 
		{
			for (int dj = 0; dj < sample_height; dj++) 
			{
				double value = sample(di, dj);
				int col = i * sample_width  + di;
				int row = j * sample_height + dj; 
				map(row, col) = value;
			}
		}
	}
	return map;
}
