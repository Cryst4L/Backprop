#include "Layer.h"

using namespace Backprop;

VectorXd Layer::getUnshapedParameters() 
{
	return VectorXd(0);
}

VectorXd Layer::getUnshapedGradient() 
{
	return VectorXd(0);
}

void Layer::setUnshapedParameters(VectorXd& unshaped)
{
	unshaped = VectorXd(0);
}

void Layer::setUnshapedGradient(VectorXd& unshaped)
{
	unshaped = VectorXd(0);
}


