#pragma once
#include <Eigen/Core>
#include "Linear.h"
#include "Activation.h"

namespace Backprop
{
class Network 
{
  private:
	std::vector<Layer*> _layer_stack;
	std::vector<VectorXd> _input_stack;

  public:
	VectorXd propagate(VectorXd& sample);
	VectorXd backpropagate(VectorXd& epsilon);

	VectorXd getUnshapedParameters();
	VectorXd getUnshapedGradient();

	VectorXd getLayerParameters(int layer_index);
	VectorXd getLayerGradient(int layer_index);

	VectorXd addLinearLayer(int input_size, int output_size);
	VectorXd addActivationLayer(int size, Activation::ActFunc activation);
	
};
}
