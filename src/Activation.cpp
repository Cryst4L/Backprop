#include "Activation.h"
#include <Eigen/Core>

using namespace Backprop;

Activation::Activation(int size, ActFunc activation)
  : _size(size), _activation(activation)
{}

bool Activation::hasParameters()
{
	return false;
}

VectorXd Activation::propagate(VectorXd& input)
{
	VectorXd output(input.size());

	switch (_activation)
	{ 
		case SIGM:
			output.array() = 1.0 / ((-input).array().exp() + 1.0);
			break;		

		case RELU:
			output.array() = input.array().max(0);
			break;
	}

	return output;
}

VectorXd Activation::backpropagate(VectorXd& input, VectorXd& epsilon)
{
	VectorXd input_grad(input.size());

	switch (_activation)
	{
		case SIGM:
//			VectorXd output = propagate(input);
//			input_grad.array() = output.array() * (1.0 - output.array());
			input_grad.array() = propagate(input).array() * (1.0 - propagate(input).array());
			break;

		case RELU:
			input_grad.array() = (input.array() > 0).cast<double>();	   
			break;
	}	

	epsilon.array() = input_grad.array() * epsilon.array();

	return epsilon;
}
