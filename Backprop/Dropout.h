#pragma once
#include <Eigen/Core>

namespace Backprop 
{
class Dropout : public Layer 
{
  private:

	double m_ratio;

	bool m_use_dropout;

	VectorXd m_mask;

  public:

	Dropout(double ratio = 0.5, bool use_dropout = true)
	: m_ratio(ratio), m_use_dropout(use_dropout)
	{}

	VectorXd forwardProp(VectorXd& input)
	{
		if (m_use_dropout == false)
			return (1 - m_ratio) * input;

		// Else ...

		int size = input.size();

		m_mask = VectorXd::Random(size);
		
		m_mask = 0.5 * (m_mask.array() + 1);

		m_mask = (m_mask.array() > m_ratio).cast <double> ();

		VectorXd output = m_mask.array() * input.array();

		//std::cout << mask << std::endl;

		return output;
	}

	VectorXd backProp(VectorXd& input, VectorXd& ein)
	{
		(void) input;

		if (m_use_dropout == false)
			return ein;
	
		// Else ...

		VectorXd eout = m_mask.array() * ein.array();

		return eout;
	}

	void enableDropout() { m_use_dropout = true; }

	void disableDropout() { m_use_dropout = false; }

	////////////////////////////////////////////////////////////////////////////

	VectorXd getParameters() { return VectorXd(0); }
	VectorXd getGradient() { return VectorXd(0); }

	void setParameters(VectorXd& parameters) { (void) parameters; }
	void setGradient(VectorXd& gradient) { (void) gradient;}

	Dropout* clone() { return new Dropout(*this); }
};
}
