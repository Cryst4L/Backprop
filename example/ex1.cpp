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


// 98.7

int main(void)
{
	srand(11);

	/// Load data //////////////////////////////////////////////////////////////

	Backprop::MNIST train_set("data/MNIST", Backprop::TRAIN);

	const int sample_size = 28;
	std::vector <MatrixXd> samples;

	for (int i = 0; i < 100; i++) {
		VectorXd sample = train_set.samples().row(i);
		MatrixXd reshaped = Map <MatrixXd> (sample.data(), sample_size, sample_size);
		samples.push_back(reshaped);
	}
	
	MatrixXd sample_map = buildMapfromData(samples, sample_size, sample_size);

	Backprop::Display sample_display(&sample_map, "Samples");

	// Setup the Net ///////////////////////////////////////////////////////////

	int kernel_size = 7;

	Backprop::Network net;

	// Multiple Conv ///////////////////////////

	net.addConvolutionalLayer(28, 28, kernel_size, 1, 5);
	net.addActivationLayer(Backprop::RELU);

	net.addLinearLayer(2420, 50);
	net.addActivationLayer(Backprop::RELU);

	///////////////////////////////////////////

	net.addLinearLayer(50, 50);
	net.addActivationLayer(Backprop::RELU);

	net.addLinearLayer(50, 10);
	net.addActivationLayer(Backprop::RELU);

	// Train the Net ///////////////////////////////////////////////////////////

	Backprop::Dataset batch;
	const int N_BATCH = 10000;
	const int BATCH_SIZE = 100;

	Backprop::Timer timer;

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
/*
	VectorXd parameters = net.layer(0).getParameters();

	std::vector <MatrixXd> kernels;

	int unshaped_size = kernel_size * kernel_size;
	
	for (int i = 0; i < 5; i++) {
		VectorXd unshaped_parameters = parameters.segment(i * unshaped_size, unshaped_size);
		MatrixXd reshaped = Map <MatrixXd> (unshaped_parameters.data(), kernel_size, kernel_size);
		kernels.push_back(reshaped);
	}
*/	
	Backprop::Convolutional* layer_p = (Backprop::Convolutional*) &(net.layer(0));
	std::vector <MatrixXd> kernels = layer_p->getKernels();

	MatrixXd kernel_map = buildMapfromData(kernels, kernel_size, kernel_size);
	Backprop::Display kernel_display(&kernel_map, "Kernels");

	/////////////////////////////////

	std::vector <MatrixXd> features;

	int feature_size = 28 - kernel_size + 1;

	VectorXd parameters = net.layer(2).getParameters();
	MatrixXd reshaped_parameters = Map <MatrixXd> (parameters.data(), 50, feature_size * feature_size);
	
	for (int i = 0; i < reshaped_parameters.rows(); i++) {
		VectorXd feature = reshaped_parameters.row(i);
		MatrixXd reshaped = Map <MatrixXd> (feature.data(), feature_size, feature_size);
		features.push_back(reshaped);
	}

	MatrixXd feature_map = buildMapfromData(features, feature_size, feature_size);
	Backprop::Display feature_display(&feature_map, "Features");

	// Test the net accuracy ///////////////////////////////////////////////////

	Backprop::MNIST test_set("data/MNIST", Backprop::TEST);

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
