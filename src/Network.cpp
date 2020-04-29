#include "Network.h"
/*
using namespace Backprop;

Network::Network()
  : _cost_function(Cost::MSE)
{}

VectorXd Network::propagate(VectorXd& input)
{
	input_stack.clear();
	input_stack.push_back(input);

	for (int i = 0; i < _layer_stack.size(); i++)
	{
		input = _layer_stack[i].propagate(input)
		input_stack.push_back(input);
	}
	
	VectorXd output = input;
	
	return output;
}

VectorXd Network::backpropagate(VectorXd& output, VectorXd& target)
{

	VectorXd sensitivity = _cost.backpropagate(output, target);


	for (int i = (_layer_stack.size()-1) ; i >= 0; i--)
		sensitivity = layer_stack[i].backpropagate(input_stack[i], sensitivity)

	return sensitivity;
}
*/ 
