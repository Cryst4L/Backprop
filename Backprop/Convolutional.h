#pragma once
#include "Layer.h"
#include "Reshape.h"

using namespace Eigen;

namespace Backprop
{

class Convolutional : public Layer
{
  private :

	int m_input_map_rows;
	int m_input_map_cols;

	int m_output_map_rows;
	int m_output_map_cols;

	int m_kernel_size;

	int m_nb_input_maps;
	int m_nb_output_maps;

	bool m_propagate_ein;

	std::vector <MatrixXd> m_input_maps;
	std::vector <MatrixXd> m_output_maps;

	std::vector <MatrixXd> m_ein_maps;
	std::vector <MatrixXd> m_eout_maps;	

	std::vector <MatrixXd> m_kernels;
	std::vector <MatrixXd> m_gradients;

	MatrixXd convOp(MatrixXd& input, MatrixXd& kernel)
	{
		int output_rows = input.rows() - kernel.rows() + 1;
		int output_cols = input.cols() - kernel.cols() + 1; 

		MatrixXd output(output_rows, output_cols);

		for (int i = 0; i < output.rows(); i++) 
			for (int j = 0; j < output.cols(); j++)
			{
				float acc = 0;

				for (int di = 0; di < kernel.rows(); di++) 
					for (int dj = 0; dj < kernel.cols(); dj++)
						acc += kernel(di, dj) * input(i + di, j + dj);

				output(i, j) = acc;
			}				

		return output;
	}

	MatrixXd deConvOp(MatrixXd& input, MatrixXd& kernel)
	{	
		int extended_rows = input.rows() + 2 * (kernel.cols() - 1);
		int extended_cols = input.cols() + 2 * (kernel.rows() - 1);

		MatrixXd extended = MatrixXd::Zero(extended_rows, extended_cols);

		int k = m_kernel_size;

		extended.block(k - 1, k - 1, input.rows(), input.cols()) = input;

		MatrixXd tranposed_kernel = kernel.transpose();
		
		MatrixXd output = convOp(extended, tranposed_kernel);

		return output;
	}

	MatrixXd computeGradient(MatrixXd& input, MatrixXd& ein)
	{
		MatrixXd gradient(m_kernel_size, m_kernel_size);

		for (int i = 0; i < gradient.rows(); i++)
			for (int j = 0; j < gradient.cols(); j++)
			{
				MatrixXd block = input.block(i, j, ein.rows(), ein.cols());
				gradient(i, j) = (ein.array() * block.array()).sum();
			}
/*
		for (int i = 0; i < gradient.rows(); i++) 
			for (int j = 0; j < gradient.cols(); j++)
			{
				double acc = 0;

				for (int di = 0; di < ein.rows(); di++) 
					for (int dj = 0; dj < ein.cols(); dj++)
						acc += ein(di, dj) * input(i + di, j + dj);

				gradient(i, j) = acc;
			}	
*/
		return gradient;
	}


  public :

	Convolutional(int input_map_rows, int input_map_cols, 
	              int kernel_size, int nb_input_maps, 
	              int nb_output_maps, bool propagate_ein = 1)
	 : m_input_map_rows(input_map_rows), 
	   m_input_map_cols(input_map_cols), 
	   m_output_map_rows(input_map_rows - kernel_size + 1), 
	   m_output_map_cols(input_map_cols - kernel_size + 1),
	   m_kernel_size(kernel_size), 
	   m_nb_input_maps(nb_input_maps), 
	   m_nb_output_maps(nb_output_maps),
	   m_propagate_ein(propagate_ein)
	{
		int nb_kernel = nb_input_maps * nb_output_maps;

		for (int i = 0; i < nb_kernel; i++)
			m_kernels.push_back(MatrixXd(kernel_size, kernel_size));

		for (int i = 0; i < nb_kernel; i++)
			m_gradients.push_back(MatrixXd(kernel_size, kernel_size));

		for (int i = 0; i < m_nb_input_maps; i++)
			m_input_maps.push_back(MatrixXd(m_input_map_rows, m_input_map_cols));

		for (int i = 0; i < m_nb_output_maps; i++)
			m_output_maps.push_back(MatrixXd(m_output_map_rows, m_output_map_cols));

		for (int i = 0; i < m_nb_output_maps; i++)
			m_ein_maps.push_back(MatrixXd(m_output_map_rows, m_output_map_cols));

		for (int i = 0; i < m_nb_input_maps; i++)
			m_eout_maps.push_back(MatrixXd(m_input_map_rows, m_input_map_cols));

		initParameters();
	}

	void initParameters()
	{
		for (int n = 0; n < m_nb_input_maps * m_nb_output_maps; n++) {
			m_kernels[n].setRandom();
			m_kernels[n] *= 1 / (float) (m_kernel_size * std::sqrt(m_nb_input_maps));
		}
	}

	VectorXd forwardProp(VectorXd& unshaped_inputs)
	{
		// Setup the maps
		reshape(unshaped_inputs, m_input_maps);

		for (int j = 0; j < m_nb_output_maps; j++)
			m_output_maps[j].setZero();

		// Perform the convolutions
		for (int i = 0; i < m_nb_input_maps; i++)
			for (int j = 0; j < m_nb_output_maps; j++)
				m_output_maps[j] += convOp(m_input_maps[i], m_kernels[i * m_nb_output_maps + j]); // HERE !
		
		// Unshape the data and return
		VectorXd unshaped_outputs;
		unshape(m_output_maps, unshaped_outputs);

		return unshaped_outputs;
	}	

	// TODO Reset the Gradient here
	VectorXd backProp(VectorXd& unshaped_inputs, VectorXd& unshaped_eins)
	{
		// Reshape the inputs
		reshape(unshaped_inputs, m_input_maps);
		reshape(unshaped_eins, m_ein_maps);

		// Compute the gradients
		for (int i = 0; i < m_nb_input_maps; i++)
			for (int j = 0; j < m_nb_output_maps; j++)
				m_gradients[i * m_nb_output_maps + j] = computeGradient(m_input_maps[i], m_ein_maps[j]);

		// Return if we dont compute eouts
		if (!m_propagate_ein)
			return VectorXd(0);

		// Else, setup the eout maps
		for (int i = 0; i < m_nb_input_maps; i++)
			m_eout_maps[i].setZero();		
			
		// Perform the deconvolutions
		for (int i = 0; i < m_nb_input_maps; i++)
			for (int j = 0; j < m_nb_output_maps; j++)
				m_eout_maps[i] += deConvOp(m_ein_maps[j], m_kernels[i * m_nb_output_maps + j]); // HERE !

		// Unshape eout and return
		VectorXd unshaped_eouts;
		unshape(m_eout_maps, unshaped_eouts);
		return unshaped_eouts;
	}

	VectorXd getParameters()
	{
		VectorXd unshaped_kernels;

		unshape(m_kernels, unshaped_kernels);

		return unshaped_kernels;
	}

	VectorXd getGradient()
	{
		VectorXd unshaped_gradients;

		unshape(m_gradients, unshaped_gradients);

		return unshaped_gradients;
	}

	void setParameters(VectorXd& parameters)
	{
		reshape(parameters, m_kernels);
	}

	void setGradient(VectorXd& gradient)
	{
		reshape(gradient, m_gradients);
	}

	Convolutional* clone() 
	{ 
		return new Convolutional(*this); 
	}

	std::vector <MatrixXd> getKernels()
	{
		return m_kernels;
	}


};
}

