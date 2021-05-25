#pragma once
#include <Eigen/Core>

namespace Backprop 
{
enum IO {I, H, O};

class Layer 
{
  public:

	virtual VectorXd forwardProp(VectorXd& input) = 0;
	virtual VectorXd backProp(VectorXd& input, VectorXd& ein) = 0;

	virtual VectorXd getParameters() = 0;
	virtual VectorXd getGradient() = 0;

	virtual void setParameters(VectorXd& parameters) = 0;
	virtual void setGradient(VectorXd& gradient) = 0;

	virtual Layer* clone() = 0;

	virtual ~Layer() {}

};
}
