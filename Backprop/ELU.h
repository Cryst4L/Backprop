#pragma once
#include "Layer.h"

using namespace Eigen;
using namespace std;

namespace Backprop
{
class ELU : public Layer
{
  private : 

	int m_size;	

	double m_alpha;
	double m_alpha_grad;

  public :
	
	ELU(int size)
    : m_size(size)
	{
		initParameters();
	}

	void initParameters()
	{
		m_alpha = 0.5;
	}	

	VectorXd forwardProp(VectorXd& input)
	{
		VectorXd output(m_size);

		m_alpha = max(m_alpha, 0.01);
	
		double a = m_alpha;

		for (int i = 0; i < m_size; i++)
			output(i) = (input(i) < 0) ? a * (exp(input(i) / a) - 1) : input(i);	

		return output;
	}

	VectorXd backProp(VectorXd& input, VectorXd& ein)
	{
		m_alpha = max(m_alpha, 0.01);

		double a = m_alpha;

		// Compute the gradient wrt the parameter
		m_alpha_grad = 0;
		for (int i = 0; i < m_size; i++) { 
			if (input(i) < 0)
				m_alpha_grad += ein(i) * (exp(input(i) / a) * (1 - input(i) / a) - 1);
		}

		// Propagate the ein
		VectorXd eout(m_size);
		for (int i = 0; i < m_size; i++) {	
			if (input(i) < 0) {	
				eout(i) = ein(i) * exp(input(i) / a);
			} else {
				eout(i) = ein(i);
			}
		}
		
		return eout;
	}	

	VectorXd getParameters()
	{
		VectorXd parameter(1);
		parameter(0) = m_alpha;
		return parameter;
	}

	VectorXd getGradient()
	{
		VectorXd gradient(1);
		gradient(0) = m_alpha_grad;
		return gradient;
	}

	void setParameters(VectorXd& parameters)
	{
		m_alpha = parameters(0);
	}

	void setGradient(VectorXd& gradient)
	{
		m_alpha_grad = gradient(0);
	} 


	ELU* clone() { return new ELU(*this); }

	///////////////////////////////////////////////////////////////////////////

};
}
