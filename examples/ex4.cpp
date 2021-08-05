// ConvNet + Dropout

#include "Backprop/Core.h"

using namespace Backprop;

// 98.83 s0 | 98.86 s9 | 98.90 s10 | 99.05 s11

int main(void)
{
	srand(0); 

	/// Load the dataset ///////////////////////////////////////////////////////

	MNIST train_set("data/MNIST", TRAIN);

	/// Display some data //////////////////////////////////////////////////////

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

	net.addConvolutionalLayer(28, 28, 7, 1, 16, 0);
	net.addActivationLayer(RELU);
	net.addDropoutLayer(0.25);

	net.addSamplingLayer(22, 22, 16, 2);

	net.addConvolutionalLayer(11, 11, 5, 16, 16);
	net.addActivationLayer(RELU);
	net.addDropoutLayer(0.25);

	net.addLinearLayer(784, 50);
	net.addActivationLayer(RELU);
	net.addDropoutLayer(0.25);

	net.addLinearLayer(50, 10);
	net.addActivationLayer(RELU);

	// Train the Net ///////////////////////////////////////////////////////////

	Dataset batch;
	const int N_BATCH = 100;
	const int BATCH_SIZE = 1000;

	VectorXd train_costs = VectorXd::Zero(N_BATCH);
	Plot plot(&train_costs, "train cost", "red");

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

		train_costs(n) = SGD(net, batch, SSE, 0.01);
		std::cout << "Average cost = " << train_costs(n) << std::endl;

		// Plot the training progress //////////////////////////////////////////////

		plot.render();
	}

	float elapsed = timer.getTimeSec();
	std::cout << " elapsed time : " << elapsed << "sec" << std::endl; 

	// Display the features ////////////////////////////////////////////////////

	Convolutional* conv_layer_p = (Convolutional*) &(net.layer(0));
	std::vector <MatrixXd> kernels = conv_layer_p->getKernels();
	int kernel_size = kernels[0].rows();

	MatrixXd kernel_map = buildMapfromData(kernels, kernel_size, kernel_size);
	Display kernel_display(&kernel_map, "Kernels");

	/////////////////////////////////

	Convolutional* conv_layer_2_p = (Convolutional*) &(net.layer(4));
	kernels = conv_layer_2_p->getKernels();
	kernel_size = kernels[0].rows();

	MatrixXd kernel_map_2 = buildMapfromData(kernels, kernel_size, kernel_size);
	Display kernel_display_2(&kernel_map_2, "Kernels");

	/////////////////////////////////

	Linear* lin_layer_p = (Linear*) &(net.layer(7));
	std::vector <VectorXd> unshaped_features = lin_layer_p->getFeatures();

	int feature_size = 7;
	int nb_features_per_row = 16;
	
	std::vector <MatrixXd> features;
	for (int i = 0; i < (int) unshaped_features.size(); i++)
	{
		for (int j = 0; j < nb_features_per_row; j++) 
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

	net.disableDropout();

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


