#pragma once
#include <Eigen/Core>

namespace Backprop
{
using namespace Eigen;

class Layer
{
  public:	
	virtual VectorXd propagate(VectorXd& input)=0;
	virtual VectorXd backpropagate(VectorXd& input, VectorXd& epsilon)=0;

	virtual bool hasParameters()=0;

	virtual VectorXd getUnshapedParameters();
	virtual VectorXd getUnshapedGradient();

	virtual void setUnshapedParameters(VectorXd& unshaped);
	virtual void setUnshapedGradient(VectorXd& unshaped);
};
}
