// PCFD loader
///////////////////////////////////////////////////////////////////////////////
#pragma once

#include <iostream>
#include <fstream>
#include <Eigen/Core>

using namespace Eigen;

namespace Backprop
{

class PCFD
{
  private:

	int m_nb_samples;
	int m_sample_size;
	int m_sample_length;

	MatrixXd m_samples;

  public:

	PCFD(const char * file_path)
	{
		m_sample_size   = 52;
		m_nb_samples    = 5000;
		m_sample_length = m_sample_size * m_sample_size;

		m_samples = MatrixXd(m_nb_samples, m_sample_length);

		std::ifstream file(file_path);

		if (!file.is_open()) {
			std::cout << "Could not open '" << file_path << "'" << std::endl;
			return;
		}		

		char * buffer = new char [m_nb_samples * m_sample_length];

		file.read(buffer, m_nb_samples * m_sample_length);

		file.close();

		for (int i = 0; i < m_nb_samples; i++)
			for (int j = 0; j < m_sample_length; j++)
				m_samples(i, j) = (unsigned char) buffer[i * m_sample_length + j] / 256.f; 

		delete [] buffer;
	}

	MatrixXd& samples() { return m_samples; }

	int size() { return m_nb_samples; }

	int sampleSize() { return m_sample_size; }
};
}

