#include "Linear.h"
#include <Eigen/Core>

using namespace Backprop;

Linear::Linear(int input_size, int output_size)
  : _input_size(input_size), _output_size(output_size),
	_parameters(output_size, input_size), _gradient(output_size, input_size),
	_bias_parameters(output_size),_bias_gradient(output_size)
{}

VectorXd Linear::propagate(VectorXd& input) 
{
	return _parameters * input + _bias_parameters;
}

VectorXd Linear::backpropagate(VectorXd& input, VectorXd& sensitivity) 
{
	_gradient = sensitivity *  input.transpose();
	_bias_gradient = sensitivity;

	return sensitivity * _parameters;
}

bool Linear::hasParameters() 
{
	return true;
}

VectorXd Linear::getUnshapedParameters() 
{
	VectorXd unshaped((_input_size + 1) * _output_size);
	unshaped << _parameters.array(), _bias_parameters;
	return unshaped;
}

VectorXd Linear::getUnshapedGradient() 
{
	VectorXd unshaped((_input_size + 1) * _output_size);
	unshaped << _gradient.array(), _bias_gradient;
	return unshaped;
}

void Linear::setUnshapedParameters(VectorXd& unshaped)
{
	_parameters.array() = unshaped.head(_input_size * _output_size).array();
	_bias_parameters.array() = unshaped.tail(_output_size).array();
}

void Linear::setUnshapedGradient(VectorXd& unshaped)
{
	_gradient.array() = unshaped.head(_input_size * _output_size).array();
	_bias_gradient.array() = unshaped.tail(_output_size).array();
}

