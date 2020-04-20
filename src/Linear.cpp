#include "Linear.h"
#include <Eigen/Core>

using namespace Backprop;

Linear::Linear(int input_size, int output_size, bool hasBias)
  : _input_size(input_size), _output_size(output_size),
	_parameters(output_size, input_size), _gradient(output_size, input_size)
	_bias_parameters(output_size), bias_gradient(output_size)
{}

VectorXd Linear::propagate(VectorXd& input) {
	return _parameters * input + _bias_parameters;
}

VectorXd Linear::backpropagate(VectorXd& input, VectorXd& epsilon) {
	_gradient = epsilon *  input.transpose();
	return epsilon * _parameters;
}

bool Linear::hasParameters() {
	return true;
}

VectorXd Linear::getUnshapedParameters() {
	return _parameters.array();
}

VectorXd Linear::getUnshapedGradient() {
	return _gradient.array();
}
