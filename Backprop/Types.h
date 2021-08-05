#pragma once
#include <Eigen/Core>

using namespace Eigen;

namespace Backprop
{
typedef unsigned char BYTE;

enum DatasetType {TRAIN, TEST, VALIDATION};

typedef struct {
	std::vector <VectorXd> inputs;
	std::vector <VectorXd> targets;
} Dataset;

}
