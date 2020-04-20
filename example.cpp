#include <iostream>
#include "Eigen/Core"
#include "src/Layer.h"
#include "src/Activation.h"

using namespace Eigen;
using namespace Backprop;

int main(void)
{
/*
	VectorXd a(5);
	VectorXd b(5);
	a << 1., 2., 3., 4., 5.;
	b = a;
	MatrixXd m = b * a.transpose();
	std::cout << m << std::endl;
*/
	//////////////////////////// 

	VectorXd a(5);
	a << 1., 2., 3., 4., 5.;
	Activation f(5, Activation::SIGM);
	a = f.propagate(a);
	std::cout << a << std::endl;

	///////////////////////////

	VectorXd a(5);
	a << 1., 2., 3., 4., 5.;

	return 0;
}
