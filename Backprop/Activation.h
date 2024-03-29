#pragma once
#include <Eigen/Core>
#include <Backprop/Layer.h>

namespace Backprop 
{
enum ActivationEnum {SIGM, TANH, RELU, SRELU, LIN};

class Activation : public Layer 
{
  private:

	ActivationEnum m_function;

  public:

	Activation(ActivationEnum function)
	: m_function(function)
	{}

	VectorXd forwardProp(VectorXd& input)
	{
		return actFun(input);
	}

	VectorXd backProp(VectorXd& input, VectorXd& ein)
	{
		return ein.cwiseProduct(actGrad(input));
	}

	VectorXd actFun(VectorXd& input)
	{
		VectorXd output(input.size());

		switch(m_function) {
			case SIGM :
				output.array() = 1.0 / (1.0 + (-input).array().exp());
				break;

			case TANH :
				output.array() = (input.array().exp() - (-input).array().exp())
				               / (input.array().exp() + (-input).array().exp()); 
				break;

			case RELU :
				output.array() = input.array().max(0);
				break;

			case SRELU :
				output.array() = (1 + input.array().exp()).log();	
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

			case TANH :
				output.array() = 1 - actFun(input).array() * actFun(input).array();
				break;

			case RELU :
				output.array() = (input.array() > 0).cast<double>();
				break;
			
			case SRELU:
				output.array() = 1.0 / (1.0 + (-input).array().exp());
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
