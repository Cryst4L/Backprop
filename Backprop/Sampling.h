#pragma once
#include "Layer.h"
#include "Reshape.h"

using namespace Eigen;

namespace Backprop
{
class Sampling : public Layer
{
  private : 

	int m_input_map_rows;
	int m_input_map_cols;

	int m_output_map_rows;
	int m_output_map_cols;

	int m_nb_maps;
	int m_ratio;

	std::vector <MatrixXd> m_input_maps;
	std::vector <MatrixXd> m_output_maps;

	std::vector <MatrixXd> m_ein_maps;
	std::vector <MatrixXd> m_eout_maps;

	void subsample(MatrixXd& input, MatrixXd& output, int ratio)
	{
		double coeff = 1 / (double) (ratio * ratio);

		for (int i = 0; i < output.rows(); i++)
			for (int j = 0; j < output.cols(); j++)
			{
				double acc = 0;

				for (int di = 0; di < ratio; di++)
					for (int dj = 0; dj < ratio; dj++)
						acc += input(i * ratio + di, j * ratio + dj);
				
				output(i, j) = acc * coeff;
			}
	}

	void upsample(MatrixXd& input, MatrixXd& output, int ratio)
	{
		double coeff = 1 / (double) (ratio * ratio);

		for (int i = 0; i < input.rows(); i++)
			for (int j = 0; j < input.cols(); j++)
			{
				for (int di = 0; di < ratio; di++)
					for (int dj = 0; dj < ratio; dj++)
						output(i * ratio + di, j * ratio + dj) = input(i, j) * coeff;
			}
	}

  public :

	Sampling(int input_map_rows, int input_map_cols, int nb_maps, int ratio)
	 : m_input_map_rows(input_map_rows), 
	   m_input_map_cols(input_map_cols),
	   m_output_map_rows(input_map_rows / ratio), 
	   m_output_map_cols(input_map_cols / ratio),  
	   m_nb_maps(nb_maps), m_ratio(ratio)
	{
		for (int i = 0; i < m_nb_maps; i++)
			m_input_maps.push_back(MatrixXd(m_input_map_rows, m_input_map_cols));

		for (int i = 0; i < m_nb_maps; i++)
			m_output_maps.push_back(MatrixXd(m_output_map_rows, m_output_map_cols));

		for (int i = 0; i < m_nb_maps; i++)
			m_ein_maps.push_back(MatrixXd(m_output_map_rows, m_output_map_cols));

		for (int i = 0; i < m_nb_maps; i++)
			m_eout_maps.push_back(MatrixXd(m_input_map_rows, m_input_map_cols));
	}

	VectorXd forwardProp(VectorXd& unshaped_inputs)
	{
		// Reshape the inputs
		reshape(unshaped_inputs, m_input_maps);

		// Subsample
		for (int n = 0; n < m_nb_maps; n++)
			subsample(m_input_maps[n], m_output_maps[n], m_ratio);

		// Unshape the outputs
		VectorXd unshaped_output;
		unshape(m_output_maps, unshaped_output);

		return unshaped_output;
	}

	VectorXd backProp(VectorXd& unshaped_inputs, VectorXd& unshaped_eins)
	{
		// Reshape the inputs
		reshape(unshaped_eins, m_ein_maps);

		// Upsample
		for (int n = 0; n < m_nb_maps; n++)
			upsample(m_ein_maps[n], m_eout_maps[n], m_ratio);

		// Unshape the outputs
		VectorXd unshaped_eouts;
		unshape(m_eout_maps, unshaped_eouts);

		// Prevent the compiler from complaining
		(void) unshaped_inputs;

		return unshaped_eouts;
	}

	VectorXd getParameters() { return VectorXd(0); }
	VectorXd getGradient() { return VectorXd(0); }

	void setParameters(VectorXd& parameters) { (void) parameters; }
	void setGradient(VectorXd& gradient) { (void) gradient;}

	Sampling* clone() { return new Sampling(*this); }

};
}
