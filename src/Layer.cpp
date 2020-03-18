#include "Layer.h"

bool Layer::hasParameters() {
	return true;
}

Tenso1D Layer::getUnshapedParameters() {
	return Tensor1D(0);
}

