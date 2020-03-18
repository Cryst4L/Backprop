#include "Linear.h"
#include "Tensor2D.h"

Linear::Linear(int input_size, int output_size)
  : _input_size(input_size), 
    _output_size(output_size),
	_parameters(output_size, input_size),
    _gradient(output_size, input_size)
{}

Tensor1D Linear::propagate(Tensor1D& input) {
	return _parameters * input;
}

Tensor1D Linear::backpropagate(Tensor1D& input, Tensor1D& epsilon) {
	_gradient = outer(epsilon, input);
	return epsilon * _parameters;
}

bool Linear::hasParameters() {
	return true;
}

Tensor1D Linear::getUnshapedParameters() {
	return _parameters.flat();
}

