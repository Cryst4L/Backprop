// Autoencoder with TANH

#include "Backprop/Core.h"

using namespace Backprop;

int main(void)
{
	srand(0);

	/// Load data //////////////////////////////////////////////////////////////

	PCFD dataset("data/PCFD.dat");

	/// Display some data //////////////////////////////////////////////////////

	std::vector <MatrixXd> samples;
	const int sample_size = dataset.sampleSize();

	for (int i = 0; i < 100; i++) {
		VectorXd sample = dataset.samples().row(i);
		MatrixXd reshaped = Map <MatrixXd> (sample.data(), sample_size, sample_size);
		samples.push_back(reshaped);
	}
	
	MatrixXd sample_map = buildMapfromData(samples, sample_size, sample_size);
	Display sample_display(&sample_map, "Samples");

	// Setup the Net ///////////////////////////////////////////////////////////

	Network net;

	net.addLinearLayer(2704, 200);
	net.addActivationLayer(SIGM);

	net.addLinearLayer(200, 50);
	net.addActivationLayer(SIGM);

	net.addLinearLayer(50, 200);
	net.addActivationLayer(SIGM);

	net.addLinearLayer(200, 2704);
	net.addActivationLayer(SIGM);

	// Train the Net ///////////////////////////////////////////////////////////

	Dataset batch;
	const int N_BATCH = 200;
	const int BATCH_SIZE = 100;

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
			int s = rand() % dataset.samples().rows();
			VectorXd sample = dataset.samples().row(s);

			batch.inputs.push_back(sample);
			batch.targets.push_back(sample);
		}

		// Train the net on the batch //////////////////////////////////////////

		train_costs(n) = SGD(net, batch, SSE, 0.01);
		std::cout << "Average cost = " << train_costs(n) << std::endl;

		// Plot the training progress //////////////////////////////////////////

		plot.render();
	}

	float elapsed = timer.getTimeSec();
	std::cout << " elapsed time : " << elapsed << "sec" << std::endl; 

	// Display the features ////////////////////////////////////////////////////

	int feature_size = sample_size;
	std::vector <MatrixXd> features;

	VectorXd parameters = net.layer(0).getParameters();
	MatrixXd reshaped_parameters = Map <MatrixXd> (parameters.data(), 200, feature_size * feature_size);
	
	for (int i = 0; i < reshaped_parameters.rows(); i++) {
		VectorXd feature = reshaped_parameters.row(i);
		MatrixXd reshaped = Map <MatrixXd> (feature.data(), feature_size, feature_size);
		features.push_back(reshaped);
	}

	MatrixXd feature_map = buildMapfromData(features, feature_size, feature_size);
	Display feature_display(&feature_map, "Features");

	// Display the reconstructed faces /////////////////////////////////////////

	std::vector <MatrixXd> reconstructeds;

	for (int i = 0; i < 100; i++) {
		VectorXd sample = dataset.samples().row(i);
		sample = net.forwardPropagate(sample);
		MatrixXd reshaped = Map <MatrixXd> (sample.data(), sample_size, sample_size);
		reconstructeds.push_back(reshaped);
	}

	sample_map = buildMapfromData(reconstructeds, sample_size, sample_size);
	Display reconstructed_display(&sample_map, "Samples");

	return 0;
}

