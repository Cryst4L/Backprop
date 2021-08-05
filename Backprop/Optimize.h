#pragma once
#include "Types.h"
#include "Network.h"
#include "Cost.h"

//#include "omp.h"

using namespace Eigen;

namespace Backprop
{

double SGD(Network& network, Dataset& dataset, CostEnum cost_function = SSE, double learning_rate = 1E-2)
{
	Cost cost(cost_function);

	double average_cost = 0;

	int n = dataset.inputs.size();

	VectorXd prediction, eout;
	VectorXd input, target;
	VectorXd parameters, gradient;	

	int i;
/*	
	omp_set_num_threads(8);
	#pragma omp parallel for schedule(dynamic) \
	private(i, input, target, prediction, eout, parameters, gradient) \
	shared(n, average_cost, dataset, network)
*/
	for (i = 0; i < n; i++) 
	{
		//std::cout << i << " ";
		//std::cout << omp_get_thread_num() << std::endl;

		// Setup the samples
		input  = dataset.inputs[i];
		target = dataset.targets[i];

		// Compute the prediction
		prediction = network.forwardPropagate(input);

		average_cost += cost.computeCost(prediction, target) / n;

		// Compute the gradient
		eout = cost.computeGradient(prediction, target);

		network.backPropagate(eout);

		// Perform the gradient descent
		parameters = network.getParameters();
		gradient   = network.getGradient();

		parameters -= learning_rate * gradient;
		network.setParameters(parameters);	
	}

/*
		for(int j = 0; j < network.nbLayers(); j++)
		{
			VectorXd parameters = network.layer(j).getParameters();
			VectorXd gradient   = network.layer(j).getGradient();
			parameters -= learning_rate * gradient;
			network.layer(j).setParameters(parameters);
		}
*/
	return average_cost;
}

}
