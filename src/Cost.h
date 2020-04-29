#pragma once
#include <Eigen/Core>

namespace Backprop
{
using namespace Eigen;

class Cost
{
  public:

	enum CostFunc {MSE, MAE /*, SMAX , CE*/};

	Cost(CostFunc cost);

	double propagate(VectorXd& output, VectorXd& target);

	VectorXd backpropagate(VectorXd& output, VectorXd& target);

  private:
	CostFunc _cost;

};

}


