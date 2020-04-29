#include <iostream>
#include "Eigen/Core"
#include "src/Layer.h"
#include "src/Display.h"
#include "src/MNIST.h"
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
	///////////////////////////
/*
	MatrixXd test = MatrixXd::Random(50,50);

	//Display(test, "random matrix");

	Display display(test, "Random Matrix");

	for (int i = 0; i < 100; i++)
	{
		test = MatrixXd::Random(50,50);
		display.render();
	}
*/
	//////////////////////////// 
/*
	MNIST train_set("../data");

	MatrixXd digits = train_set.samples();
	VectorXd example = digits.row(50);

	MatrixXd reshaped(28,28);
	for (int i = 0; i < 28; i++)
		for (int j = 0; j < 28; j++)
			reshaped(i,j) = example(28 * i + j);

	Display(reshaped, "Sample 50");
*/
	//////////////////////////// 

	VectorXd a(5);
	a << 1., 2., 3., 4., 5.;
	Activation f(5, Activation::SIGM);
	a = f.propagate(a);
	std::cout << a << std::endl;

	return 0;
}
