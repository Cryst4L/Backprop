#include <iostream>
#include <vector>
#include <cstdlib>

#include "Backprop/MNIST.h"
#include "Backprop/Display.h"
#include "Backprop/Utils.h"
#include "Backprop/Timer.h"
#include "Backprop/Types.h"
#include "Backprop/Network.h"
#include "Backprop/Convolutional.h"
#include "Backprop/Optimize.h"

using namespace Backprop;

// 98.2

int main(void)
{
	srand(9);

	/// Load data //////////////////////////////////////////////////////////////

	MNIST train_set("data/MNIST", TRAIN);

	const int sample_size = 28;
	std::vector <MatrixXd> samples;

	for (int i = 0; i < 100; i++) {
		VectorXd sample = train_set.samples().row(i);
		MatrixXd reshaped = Map <MatrixXd> (sample.data(), sample_size, sample_size);
		samples.push_back(reshaped);
	}
	
	MatrixXd sample_map = buildMapfromData(samples, sample_size, sample_size);

	Display sample_display(&sample_map, "Samples");

	// Setup the Net ///////////////////////////////////////////////////////////

	Network net;

	// Multiple Conv ///////////////////////////

	net.addConvolutionalLayer(28, 28, 9, 1, 12);
	net.addActivationLayer(RELU);

	net.addConvolutionalLayer(20, 20, 5, 12, 8);
	net.addActivationLayer(RELU);

	net.addLinearLayer(2048, 50);
	net.addActivationLayer(RELU);

	///////////////////////////////////////////
/*
	net.addLinearLayer(50, 50, H);
	net.addActivationLayer(50, RELU);
*/
	net.addLinearLayer(50, 10);
	net.addActivationLayer(RELU);

	// Train the Net ///////////////////////////////////////////////////////////

	Dataset batch;
	const int N_BATCH = 1000;
	const int BATCH_SIZE = 100;

	Timer timer;

	for (int n = 0; n < N_BATCH; n++)
	{
		// Build a batch ///////////////////////////////////////////////////////
		
		batch.inputs.clear();
		batch.targets.clear();
		
		for (int i = 0; i < BATCH_SIZE; i++)
		{
			int s = rand() % train_set.samples().rows();

			VectorXd sample = train_set.samples().row(s);
			batch.inputs.push_back(sample);

			VectorXd target = train_set.targets().row(s);
			batch.targets.push_back(target);
		}

		// Train the net on the batch //////////////////////////////////////////

		SGD(net, batch, 0.01);
	}

	float elapsed = timer.getTimeSec();
	std::cout << " elpased time : " << elapsed << "sec" << std::endl; 

	// Display the features ////////////////////////////////////////////////////

	Convolutional* conv_layer_p = (Convolutional*) &(net.layer(0));
	std::vector <MatrixXd> kernels = conv_layer_p->getKernels();
	int kernel_size = kernels[0].rows();

	MatrixXd kernel_map = buildMapfromData(kernels, kernel_size, kernel_size);
	Display kernel_display(&kernel_map, "Kernels");

	/////////////////////////////////

	Convolutional* conv_layer_2_p = (Convolutional*) &(net.layer(2));
	kernels = conv_layer_2_p->getKernels();
	kernel_size = kernels[0].rows();

	MatrixXd kernel_map_2 = buildMapfromData(kernels, kernel_size, kernel_size);
	Display kernel_display_2(&kernel_map_2, "Kernels");

	/////////////////////////////////

	Linear* lin_layer_p = (Linear*) &(net.layer(4));
	std::vector <VectorXd> unshaped_features = lin_layer_p->getFeatures();

	int feature_size = 28 - (9 - 1)  - (5 - 1);
	
	std::vector <MatrixXd> features;
	for (int i = 0; i < (int) unshaped_features.size(); i++)
	{
		for (int j = 0; j < 8; j++) 
		{
			int size = feature_size * feature_size;
			VectorXd unshaped = unshaped_features[i].segment(j * size, size);
			MatrixXd reshaped = Map <MatrixXd> (unshaped.data(), feature_size, feature_size);
			features.push_back(reshaped);
		}
	}

	MatrixXd feature_map = buildMapfromData(features, feature_size, feature_size);
	Backprop::Display feature_display(&feature_map, "Features");

	// Test the net accuracy ///////////////////////////////////////////////////

	MNIST test_set("data/MNIST", TEST);

	int acc = 0;
	for (int i = 0; i < test_set.nbSamples(); i++)
	{
		VectorXd sample = test_set.samples().row(i);
		VectorXd target = test_set.targets().row(i);

		VectorXd prediction = net.forwardPropagate(sample);

		VectorXd::Index value, guess;

		target.maxCoeff(&value);
		prediction.maxCoeff(&guess);
		
		if (value == guess)
			acc++;
	}

	std::cout << " accuracy = " << acc / (float) test_set.nbSamples() << std::endl;

	return 0;
}

/*
	VectorXd v(4); 
	v << 1, 2, 3, 4;

	MatrixXd m(2, 2);
    m << 5, 6, 7, 8;

	v = Map <VectorXd> (m.data(), m.size());

	v(0) = 0;

	std::cout << m << std::endl;

	m(0,0) = 2;

	std::cout << v << std::endl;
*/
