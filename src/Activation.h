#pragma once
#include "Layer.h"

namespace Backprop
{
class Activation : public Layer 
{
  public:
	enum ActFunc {SIGM, RELU};

	Activation(int size, ActFunc activation);

	bool hasParameters();

	VectorXd propagate(VectorXd& input);
	VectorXd backpropagate(VectorXd& input, VectorXd& epsilon);

  private:
	int _size;
	ActFunc _activation;

};
}

