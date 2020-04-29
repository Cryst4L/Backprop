#include "Cost.h"

using namespace Backprop;

Cost::Cost(CostFunc cost)
  : _cost(cost) 
{}

double Cost::propagate(VectorXd& output, VectorXd& target)
{
	double value = 0;

	switch (_cost)
	{
		case MSE:
			value = (output - target).array().square().mean();
			break;

		case MAE:
			value = (output - target).array().abs().mean();
			break;
	}
	
	return value;	
}

VectorXd Cost::backpropagate(VectorXd& output, VectorXd& target)
{
	VectorXd gradient(output.size());

	switch (_cost)
	{
		case MSE:
			gradient = 2 * (output - target) / output.size();
			break;

		case MAE:
			gradient.array() = (output - target).array().sign() / output.size();
			break;
	}

	return gradient;
}

/*
		case CE:
		{
			for (int i = 0; i < target.size();
			{
				doubel e = 0;
				(target(i) == 1) ? 
					e = -log(output(i)) : e = -log(1 - output(i));
				value += e / target.size();				
			}
		}
		break;
*/	 

