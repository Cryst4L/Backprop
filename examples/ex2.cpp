// MLP with ELU + Softmax with CE

#include "Backprop/Core.h"

using namespace Backprop;

// 97.65 s0 

int main(void)
{
	srand(0);

	/// Load data //////////////////////////////////////////////////////////////

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

	net.addLinearLayer(784, 50);
	net.addELULayer(50);

	net.addLinearLayer(50, 50);
	net.addELULayer(50);

	net.addLinearLayer(50, 50);
	net.addELULayer(50);

	net.addLinearLayer(50, 10);
	net.addELULayer(10);

	net.addSoftMaxLayer();

	// Train the Net ///////////////////////////////////////////////////////////

	Dataset batch;
	const int N_BATCH = 1000;
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

		train_costs(n) = SGD(net, batch, CE, 0.01);
		std::cout << "Average cost = " << train_costs(n) << std::endl;

		// Plot the training progress //////////////////////////////////////////////

		plot.render();
	}

	float elapsed = timer.getTimeSec();
	std::cout << " elapsed time : " << elapsed << "sec" << std::endl;

	// Print the ELU coefficients //////////////////////////////////////////////

	std::cout << " ELU coefficients : \n"; 

	for (int i = 1; i < 9; i += 2)
		std::cout << net.layer(i).getParameters() << std::endl;
 
	// Display the features ////////////////////////////////////////////////////

	int feature_size = 28;
	std::vector <MatrixXd> features;

	VectorXd parameters = net.layer(0).getParameters();
	MatrixXd reshaped_parameters = Map <MatrixXd> (parameters.data(), 50, feature_size * feature_size);
	
	for (int i = 0; i < reshaped_parameters.rows(); i++) {
		VectorXd feature = reshaped_parameters.row(i);
		MatrixXd reshaped = Map <MatrixXd> (feature.data(), feature_size, feature_size);
		features.push_back(reshaped);
	}

	MatrixXd feature_map = buildMapfromData(features, feature_size, feature_size);
	Display feature_display(&feature_map, "Features");

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

