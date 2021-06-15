#pragma once
#include "Types.h"
#include "Network.h"
#include "Cost.h"

using namespace Eigen;

namespace Backprop
{

void SGD(Network& network, Dataset& dataset, CostEnum cost_function = SSE, double learning_rate = 1E-2)
{
	Cost cost(cost_function);

	double average_cost = 0;

	int n = dataset.inputs.size();

	// Loop over the dataset
	for (int i = 0; i < n; i++) {
	
		// Setup the samples
		VectorXd input  = dataset.inputs[i];
		VectorXd target = dataset.targets[i];
	
		// Compute the prediction
		VectorXd prediction = network.forwardPropagate(input);
		average_cost += cost.computeCost(prediction, target) / n;

		// Compute the gradient
		VectorXd eout = cost.computeGradient(prediction, target);
		network.backPropagate(eout);             

		// Perform the gradient descent
		for(int j = 0; j < network.nbLayers(); j++)
		{
			VectorXd parameters = network.layer(j).getParameters();
			VectorXd gradient   = network.layer(j).getGradient();

			parameters -= learning_rate * gradient;
			network.layer(j).setParameters(parameters);
		}
	}

	std::cout << "Average Cost = " << average_cost << std::endl;

	//display();
	}
}
