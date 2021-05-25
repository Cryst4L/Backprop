////////////////////////////////////////////////////////////////////////////////
// MNIST dataset loader
///////////////////////////////////////////////////////////////////////////////
#pragma once

#include <iostream>
#include <fstream>

#include <stdint.h>
#include <endian.h>

#undef Success //TODO: Fix this !
#include <Eigen/Core>

#include "Types.h"

using namespace Eigen;

namespace Backprop
{

class MNIST
{
  public:

	MNIST(std::string folder_path, DatasetType type = TRAIN, int size = -1)
	  :  m_size(size)
	{
		loadSamples(folder_path, type);
		loadLabels(folder_path, type);

		std::cout << " MNIST data loaded"
			      << "\n - type : " << (type == 0 ? "TRAIN" : "TEST") 
			      << "\n - size : " << m_size
			      << std::endl; 
	}

	// Get a reference to the loaded samples. 
	// Each row store a sample data using a row-major indexing. 
	MatrixXd& samples()
	{
		return m_samples;
	}

	// Get a reference to the loaded targets. 
	// Each row represent a sample class in a one-hot manner. 
	MatrixXd& targets()
	{
		return m_targets;
	}

	// Get the number of samples loaded
	int nbSamples()
	{
		return m_size;
	}

  private:

	int m_size;

	MatrixXd m_samples;
	MatrixXd m_targets;

	//! Rearrange and IDX 32-bit integer
	inline int reverseInt(int i)
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

	//! Load the MNIST samples
	void loadSamples(std::string folder_path, DatasetType type)
	{
		std::string path = (type == TRAIN) ? 
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

		if (m_size < 0) 
			m_size = reverseInt(items);

		m_samples.resize(m_size, rows * cols);

		unsigned char temp = 0;
		for(int n = 0; n < m_size; n++)
		{			
			for(int r = 0; r < rows; r++)
				for(int c = 0; c < cols; c++)
				{
					sample_file.read((char*) &temp, sizeof(temp));
					double value = (double) temp;
					m_samples(n, r * cols + c) = value / 255.0; // scale to [0:1]
				}
		}
	}

	//! Load the MNIST targets
	void loadLabels(std::string folder_path, DatasetType type)
	{
		std::string path = (type == TRAIN) ?
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

		int n_targets = 10;

		if (m_size < 0) 
			m_size = reverseInt(items);

		m_targets.resize(m_size, n_targets);

		unsigned char temp = 0;
		for (int n = 0; n < m_size; n++)
		{
			label_file.read((char*) &temp, sizeof(temp));
			for (int l = 0; l < n_targets; l++)
				m_targets(n, l) = (int) temp == l ? 1.0 : 0.0;
		}
	}

};
}



