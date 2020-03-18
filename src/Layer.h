#pragma once
#include "Tensor1D.h"

class Layer
{
  public:	
	virtual Tensor1D propagate(Tensor1D& input)=0;
	virtual Tensor1D backpropagate(Tensor1D& input, Tensor1D& epsilon)=0;

	virtual bool hasParameters()=0;
	virtual Tensor1D getUnshapedParameters()=0;

};

