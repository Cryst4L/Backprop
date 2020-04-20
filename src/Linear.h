#pragma once
#include "Layer.h"
#include <Eigen/Core>

namespace Backprop
{
class Linear : public Layer 
{
  private: 
	int _input_size;
	int _output_size;

	double _bias;

	MatrixXd _parameters;
	MatrixXd _gradient;

	VectorXd _bias_parameters;
	VectorXd _bias_gradient;

  public:
	Linear(int input_size, int output_size); 

	VectorXd propagate(VectorXd& input);
	VectorXd backpropagate(VectorXd& input, VectorXd& epsilon);

	bool hasParameters();

	VectorXd getUnshapedParameters();
	VectorXd getUnshapedGradient();

};
}
