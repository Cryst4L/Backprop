#pragma once

#include "Linear.h"
#include "Convolutional.h"
#include "Activation.h"
#include "Sampling.h"
#include "SoftMax.h"
#include "Dropout.h"
#include "ELU.h"
#include "Types.h"

using namespace Eigen;

namespace Backprop
{
class Network
{
  private :
	
	std::vector <Layer*> m_layer_stack;
	std::vector <VectorXd> m_input_stack;

  public:

	Network() {}

	Network& operator=(const Network &network)
	{
		if (this != &network) 
		{
			m_input_stack = network.m_input_stack;

			// Deep clean
			for (int i = 0; i < (int) m_layer_stack.size(); i++) 
				delete m_layer_stack[i];

			m_layer_stack.clear();

			// Deep copy
			for (int i = 0; i < (int) network.m_layer_stack.size(); i++) {
				Layer * layer = network.m_layer_stack[i]->clone();
				m_layer_stack.push_back(layer);
			}
		}
		return *this;
	}

	Network(const Network &network)
	{
		*this = network; // overloaded '=' is called
	}

	~Network()
	{
		for (int i = 0; i < (int) m_layer_stack.size(); i++)
			delete m_layer_stack[i];
	}

	////////////////////////////////////////////////////////////////////////////
	
	Layer& layer(int i) 
	{
		return *(m_layer_stack[i]);
	}

	void addLinearLayer(int input_size, int output_size, bool propagate_ein = 1)
	{
		Linear * layer = new Linear(input_size, output_size, propagate_ein);
		m_layer_stack.push_back(layer);
		m_input_stack.push_back(VectorXd(0));
	}

	void addActivationLayer(ActivationEnum function)
	{
		Activation * layer = new Activation(function);
		m_layer_stack.push_back(layer);
		m_input_stack.push_back(VectorXd(0));
	}

	void addSamplingLayer(int input_map_rows, int input_map_cols, int nb_maps, int ratio)
	{
		Sampling * layer = new Sampling(input_map_rows, input_map_cols, nb_maps, ratio);
		m_layer_stack.push_back(layer);
		m_input_stack.push_back(VectorXd(0));
	}

	void addConvolutionalLayer(int input_map_rows, int input_map_cols, 
	                           int kernel_size, int nb_input_maps, 
	                           int nb_output_maps, bool padded = 0, 
	                           bool propagate_ein = 1)
	{
		Convolutional * layer = new Convolutional(input_map_rows, input_map_cols, 
		                                          kernel_size, nb_input_maps, 
		                                          nb_output_maps, padded, 
	                                              propagate_ein);
		m_layer_stack.push_back(layer);
		m_input_stack.push_back(VectorXd(0));
	}

	void addSoftMaxLayer()
	{
		SoftMax * layer = new SoftMax();
		m_layer_stack.push_back(layer);
		m_input_stack.push_back(VectorXd(0));
	}

	void addELULayer(int size)
	{
		ELU * layer = new ELU(size);
		m_layer_stack.push_back(layer);
		m_input_stack.push_back(VectorXd(0));
	}

	void addDropoutLayer(double ratio = 0.5, bool use_dropout = true)
	{
		Dropout * layer = new Dropout(ratio, use_dropout);
		m_layer_stack.push_back(layer);
		m_input_stack.push_back(VectorXd(0));
	}

	void enableDropout()
	{
		for(int i = 0; i < (int) m_layer_stack.size(); i++) {
			Dropout * layer = dynamic_cast <Dropout*> (m_layer_stack[i]);
			if (layer != NULL)
				layer->enableDropout();
		}
	}

	void disableDropout()
	{
		for(int i = 0; i < (int) m_layer_stack.size(); i++) {
			Dropout * layer = dynamic_cast <Dropout*> (m_layer_stack[i]);
			if (layer != NULL)
				layer->disableDropout();
		}
	}

	VectorXd forwardPropagate(VectorXd& input)
	{
		VectorXd signal = input;

		for(int i = 0; i < (int) m_layer_stack.size(); i++) {
			m_input_stack[i] = signal;
			signal = m_layer_stack[i]->forwardProp(signal);
		}

		return signal;
	}

	void backPropagate(VectorXd& ein)
	{
		VectorXd signal = ein;
		int l = m_layer_stack.size();
		for(int i = l - 1; i >= 0; i--) {
			signal = m_layer_stack[i]->backProp(m_input_stack[i], signal);
		}
	}

	int nbLayers() { return m_layer_stack.size(); }
};
}
