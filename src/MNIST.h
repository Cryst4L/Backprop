#pragma once
#undef Success 
#include <Eigen/Core>

namespace Backprop
{
////////////////////////////////////////////////////////////////////////////////
// MNIST dataset loader
////////////////////////////////////////////////////////////////////////////////
class MNIST
{
  public:

	enum DatasetType {TRAIN, TEST, VALIDATION};

	MNIST(std::string folder_path, DatasetType ds_type=TRAIN, int nb_samples=-1);

	Eigen::MatrixXd& samples();

	Eigen::MatrixXd& labels();

	int NbSamples();

  private:

	int _nb_samples;

	Eigen::MatrixXd _samples;
	Eigen::MatrixXd _labels;

	inline int reverseInt(int i);

	void loadSamples(std::string folder_path, DatasetType ds_type);

	void loadLabels(std::string folder_path, DatasetType ds_type);
};
}
