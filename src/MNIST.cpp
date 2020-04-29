#include "MNIST.h"

#include <iostream>
#include <fstream>

#include <stdint.h>
#include <endian.h>

using namespace Eigen;

namespace Backprop
{
void MNIST::loadSamples(std::string folder_path, DatasetType ds_type)
{
	std::string path = (ds_type == TRAIN) ? 
		folder_path + "/train-images.idx3-ubyte" : 
		folder_path + "/t10k-images.idx3-ubyte";

	std::fstream sample_file(path.c_str(), std::ios::in | std::ios::binary);

	if (!sample_file.is_open())
	{
		std::cerr << " Failed to open '" << path << "'. Exiting ...\n";
		exit(1);
	}

	int magic, items, rows, cols;

	sample_file.read((char*) &magic, sizeof(magic));
	sample_file.read((char*) &items, sizeof(items));

	sample_file.read((char*) &rows, sizeof(rows));
	sample_file.read((char*) &cols, sizeof(cols));

	rows = reverseInt(rows);
	cols = reverseInt(cols);

	if (_nb_samples < 0) 
		_nb_samples = reverseInt(items);

	_samples.resize(_nb_samples, rows * cols);

	unsigned char temp = 0;
	for(int n = 0; n < _nb_samples; n++)
	{			
		for(int r = 0; r < rows; r++)
			for(int c = 0; c < cols; c++)
			{
				sample_file.read((char*) &temp, sizeof(temp));
				double value = (double) temp;
				_samples(n, r * cols + c) = value / 255.0; // scale to [0:1]
			}
		//std::cout << _samples.row(n).mean();
	}
}

void MNIST::loadLabels(std::string folder_path, DatasetType ds_type)
{
	std::string path = (ds_type == TRAIN) ?
		(folder_path + "/train-labels.idx1-ubyte") :
		(folder_path + "/t10k-labels.idx1-ubyte");

	std::fstream label_file(path.c_str(), std::ios::in | std::ios::binary);

	if (!label_file.is_open())
	{
		std::cerr << " Failed to open '" << path << "'\n Exiting ...\n";
		exit(1);
	}

	int magic, items;

	label_file.read((char*) &magic, sizeof(magic));
	label_file.read((char*) &items, sizeof(items));

	int n_labels = 10;

	if (_nb_samples < 0) 
		_nb_samples = reverseInt(items);

	_labels.resize(_nb_samples, n_labels);

	unsigned char temp = 0;
	for (int n = 0; n < _nb_samples; n++)
	{
		label_file.read((char*) &temp, sizeof(temp));
		for (int l = 0; l < n_labels; l++)
			_labels(n, l) = (int) temp == l ? 1.0 : 0.0;
	}
}

inline int MNIST::reverseInt(int i)
{
	unsigned char c1, c2, c3, c4;

	c1 = i & 255;
	c2 = (i >>  8) & 255;
	c3 = (i >> 16) & 255;
	c4 = (i >> 24) & 255;

	int i1, i2, i3, i4;

	i1 = (int) c1 << 24;
	i2 = (int) c2 << 16;
	i3 = (int) c3 << 8;
	i4 = (int) c4;

	return i1 + i2 + i3 + i4;
}

MNIST::MNIST(std::string folder_path, DatasetType ds_type, int nb_samples)
  :  _nb_samples(nb_samples)
{
	loadSamples(folder_path, ds_type);
	loadLabels(folder_path, ds_type);

	std::cout << " MNIST data loaded"
	          << "\n - type : " << (ds_type == 0 ? "TRAIN" : "TEST") 
	          << "\n - size : " << _nb_samples
	          << std::endl; 
}

MatrixXd& MNIST::samples()
{
	return _samples;
}

MatrixXd& MNIST::labels()
{
	return _labels;
}

int MNIST::NbSamples()
{
	return _nb_samples;
}
}
