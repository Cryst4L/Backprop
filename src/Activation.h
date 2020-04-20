#pragma once
#include "Layer.h"

namespace Backprop
{
class Activation : public Layer 
{
  private:
	int _size;

  public:

	enum ActFunc {SIGM, RELU};

	ActFunc _activation;

	Activation(int size, ActFunc activation);

	bool hasParameters();

	VectorXd propagate(VectorXd& input);
	VectorXd backpropagate(VectorXd& input, VectorXd& epsilon);
};
}

