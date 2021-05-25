#pragma once
#include <Eigen/Core>

namespace Backprop 
{

enum Function {SIGM, RELU, LIN};

class Activation : public Layer 
{
  private:

	Function m_function;

  public:
 
	Activation(Function function)
	: m_function(function)
	{}

	// Activation m_function //////////////////////////////////////////////////////////
	VectorXd forwardProp(VectorXd& input)
	{
		return actFun(input);
	}

	// Activation gradient //////////////////////////////////////////////////////////
	VectorXd backProp(VectorXd& input, VectorXd& ein)
	{
		return ein.cwiseProduct(actGrad(input));
	}

	// Activation m_function //////////////////////////////////////////////////////////
	VectorXd actFun(VectorXd& input)
	{
		VectorXd output(input.size());

		switch(m_function) {
			case SIGM :
				output.array() = 1.0 / (1.0 + (-input).array().exp());
				break;

			case RELU :
				output.array() = input.array().max(0);
				break;

			case LIN  :
				output.array() = input.array();
				break;
		}

		return output;
	}

	// Activation gradient //////////////////////////////////////////////////////////
	VectorXd actGrad(VectorXd& input)
	{
		VectorXd output(input.size());

		switch(m_function) {
			case SIGM :
				output.array() = actFun(input).array() * (1. - actFun(input).array());
				break;

			case RELU :
				output.array() = (input.array() > 0).cast<double>();
				break;

			case LIN  :
				output.setOnes();
				break;
		}

		return output;
	}

	VectorXd getParameters() { return VectorXd(0); }
	VectorXd getGradient() { return VectorXd(0); }

	void setParameters(VectorXd& parameters) { (void) parameters; }
	void setGradient(VectorXd& gradient) { (void) gradient;}

	Activation* clone() { return new Activation(*this); }
};
}
