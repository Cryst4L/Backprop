#pragma once
#include <Eigen/Core>

using namespace Eigen;

namespace Backprop
{

enum DatasetType {TRAIN, TEST, VALIDATION};

typedef struct {
	std::vector <VectorXd> inputs;
	std::vector <VectorXd> targets;
} Dataset;

}
