#pragma once
#include "Layer.h"
#include "Tensor2D.h"

class Linear : public Layer 
{
  private: 
	int _input_size;
	int _output_size;

	Tensor2D _parameters;
	Tensor2D _gradient;

  public:
	Linear(int input_size, int output_size); 

	Tensor1D propagate(Tensor1D& input);
	Tensor1D backpropagate(Tensor1D& input, Tensor1D& epsilon);

	bool hasParameters();
	Tensor1D getUnshapedParameters();

};
