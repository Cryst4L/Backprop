#include "Network.h"

VectorXd Network::propagate(VectorXd& sample) {
	for (int i = 0; i < _layer_stack.size(); i++) {
		_input_stack[i] = sample;
		sample = _layer_stack[i]->propagate(sample);
	}
	return sample;
}

VectorXd Network::backpropagate(VectorXd& epsilon) {
	for (int i = layer_stack.size()-1; i >= 0; i--) {
		epsilon = _layer_stack[i]->backpropagate(_input_stack[i], epsilon);
	}
	return epsilon;
}

VectorXd Network::getUnshapedParameters() {
	int parameters_size = 0;
	std::vector<VectorXd> parameters_stack;
	for (int i = 0; i < _layer_stack.size(); i++) {
		VectorXd layer_parameters = _layer_stack[i]->getUnshapedParameters(); 
		parameters_stack.push_back(layer_parameters);
		parameters_size += layer_parameters.size();
	}
	VectorXd network_parameters(parameters_size);
	for (int i = 0; i < parameter_stack.size(); i++) {
		network_parameters << parameters_stack[i];
	}
	return network_parameters;
}

VectorXd Network::getUnshapedGradient() {
	int gradient_size = 0;
	std::vector<VectorXd> gradient_stack;
	for (int i = 0; i < _layer_stack.size(); i++) {
		VectorXd layer_gradient = _layer_stack[i]->getUnshapedGradient(); 
		gradient_stack.push_back(layer_gradient);
		gradient_size += layer_gradient.size();
	}
	VectorXd network_gradient(gradient_size);
	for (int i = 0; i < gradient_stack.size(); i++) {
		network_gradient << gradient_stack[i];
	}
	return network_gradient;
}
