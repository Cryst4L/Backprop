#pragma once
#include "Layer.h"

using namespace Eigen;

namespace Backprop
{
class Linear : public Layer
{
  private : 

	int m_input_size;
	int m_output_size;

	bool m_propagate_ein;

	MatrixXd m_theta;
	MatrixXd m_gradient;

  public :

	Linear(int input_size, int output_size, bool propagate_ein = 1)
	: m_input_size(input_size), m_output_size(output_size), 
	  m_propagate_ein(propagate_ein)
	{
		m_theta = MatrixXd::Zero(output_size, input_size);
		m_gradient = MatrixXd::Zero(output_size, input_size);

		m_theta.setRandom();
		m_theta *= std::sqrt(6. / (input_size + output_size));
	}

	VectorXd forwardProp(VectorXd& input)
	{
		return m_theta * input;
	}

	VectorXd backProp(VectorXd& input, VectorXd& ein)
	{
		m_gradient = ein * input.transpose();

		// Return if we dont compute eouts
		if (!m_propagate_ein) return VectorXd(0);

		// Else compute eouts
		VectorXd eout = m_theta.transpose() * ein;

		return eout;
	}

	VectorXd getParameters()
	{
		VectorXd unshaped = Map <VectorXd> (m_theta.data(), m_theta.size());
		return unshaped;
	}

	VectorXd getGradient()
	{
		VectorXd unshaped = Map <VectorXd> (m_gradient.data(), m_gradient.size());
		return unshaped;
	}

	void setParameters(VectorXd& parameters)
	{
		m_theta = Map <MatrixXd> (parameters.data(), m_output_size, m_input_size);
	}

	void setGradient(VectorXd& gradient)
	{
		m_gradient = Map <MatrixXd> (gradient.data(), m_output_size, m_input_size);
	} 

	Linear* clone() { return new Linear(*this); }

	///////////////////////////////////////////////////////////////////////////

	std::vector <VectorXd> getFeatures()
	{
		std::vector <VectorXd> features;
		for (int i = 0; i < m_theta.rows(); i++)
			features.push_back(m_theta.row(i));

		return features;
	}


};
}
