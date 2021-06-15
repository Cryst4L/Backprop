#pragma once
#include <Eigen/Core>

namespace Backprop 
{
class SoftMax : public Layer 
{
  public:

	SoftMax() {}

	VectorXd forwardProp(VectorXd& input)
	{
		return softmax(input);
	}

	VectorXd backProp(VectorXd& input, VectorXd& ein)
	{
		// Naive approach: compute the Gradient first

		int size = input.size();

		MatrixXd input_grad(size, size);

		input_grad = - softmax(input) * softmax(input).transpose();

		input_grad += softmax(input).asDiagonal();

		VectorXd eout = ein.transpose() * input_grad;

		// Optimized approach: do not compute the gradient
/*
		VectorXd s = softmax(input);

		double dot_product = ein.transpose() * s;

		VectorXd eout = - dot_product * s;

		eout.array() += ein.array() * s.array(); 
*/		
		return eout;
	}

	VectorXd softmax(VectorXd& input)
	{
		VectorXd output = input.array().exp();

		double partition = output.sum();

		output /= partition;

		return output;
	}

	VectorXd getParameters() { return VectorXd(0); }
	VectorXd getGradient() { return VectorXd(0); }

	void setParameters(VectorXd& parameters) { (void) parameters; }
	void setGradient(VectorXd& gradient) { (void) gradient;}

	SoftMax* clone() { return new SoftMax(*this); }
};
}
