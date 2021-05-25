#pragma once
#include "Types.h"
#include "Network.h"

using namespace Eigen;

namespace Backprop
{

void SGD(Network& network, Dataset& dataset, double learning_rate = 1E-3, double penality = 0.)
{
	double cost = 0;

	// Loop over the dataset
	for (int i = 0; i < (int) dataset.inputs.size(); i++) {
	
		// Setup the samples
		VectorXd input  = dataset.inputs[i];
		VectorXd target = dataset.targets[i];
	
		// Compute the gradients
		VectorXd prediction = network.forwardPropagate(input);
		VectorXd error = (prediction - target);
		network.backPropagate(error);               // Fix this !


		// Accumulate the cost
		cost += error.cwiseProduct(error).sum() / (float) dataset.inputs.size(); 

		// Perform gradient descent
		for(int j = 0; j < network.nbLayers(); j++) {
			VectorXd parameters = network.layer(j).getParameters();
			parameters -= learning_rate * network.layer(j).getGradient();
			//parameters -= learning_rate * penality * parameters;
			network.layer(j).setParameters(parameters);
		}
	}

	std::cout << "Average Cost = " << cost << std::endl;

	//display();
	}
}
