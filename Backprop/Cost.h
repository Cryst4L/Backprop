#pragma once
#include <Eigen/Core>

namespace Backprop 
{
enum CostEnum {AE, SSE, CE};

static const double EPSILON = 0.001;

class Cost
{
  private:

	CostEnum m_function;

  public:

	Cost() 
	: m_function(SSE)
	{}

	Cost(CostEnum function)
	: m_function(function)
	{}

	// Activation m_function //////////////////////////////////////////////////////////
	double computeCost(VectorXd& output, VectorXd& target)
	{
		double cost = 0;
	
		switch(m_function) 
		{
			case AE :
				cost = (output - target).array().abs().sum();
				break;

			case SSE :
				cost = (output - target).array().square().sum();
				break;

			case CE :
				output = output.array().max(EPSILON).min(1 - EPSILON);
				cost = - (target.array() * output.array().log()).sum() 
                       - ((1 - target.array()) * (1 - output.array()).log()).sum();		
			break;	
		}

		return cost;
	}

	// Activation gradient //////////////////////////////////////////////////////////
	VectorXd computeGradient(VectorXd& output, VectorXd& target)
	{
		int n = output.size();

		VectorXd gradient(n);

		switch(m_function) 
		{
			case AE :
				gradient = (output - target).array().sign();
				break;

			case SSE :
				gradient = 2 * (output - target);
				break;

			case CE :
				output = output.array().max(EPSILON).min(1 - EPSILON);
				gradient = (output - target).array() 
				         / output.array() * (1 - output.array());
				break;	
		}

		return gradient;
	}
};
}
