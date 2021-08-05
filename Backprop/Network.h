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
		for(int i = l - 1; i >= 0; i--)
			signal = m_layer_stack[i]->backProp(m_input_stack[i], signal);

/*
		VectorXd signal = ein;
		int l = m_layer_stack.size();

		signal = m_layer_stack[l - 1]->backProp(m_input_stack[l - 1], signal);
		signal = m_layer_stack[l - 2]->backProp(m_input_stack[l - 2], signal);
		signal = m_layer_stack[l - 3]->backProp(m_input_stack[l - 3], signal);
		signal = m_layer_stack[l - 4]->backProp(m_input_stack[l - 4], signal);
		signal = m_layer_stack[l - 5]->backProp(m_input_stack[l - 5], signal);
		// signal = m_layer_stack[l - 6]->backProp(m_input_stack[l - 6], signal);
		//signal = m_layer_stack[l - 7]->backProp(m_input_stack[l - 7], signal);
		//signal = m_layer_stack[l - 8]->backProp(m_input_stack[l - 8], signal);
		//signal = m_layer_stack[l - 9]->backProp(m_input_stack[l - 9], signal);
*/
	}

	VectorXd getParameters() 
	{
		int size = 0;
		for(int i = 0; i < (int) m_layer_stack.size(); i++) 
			size += m_layer_stack[i]->getParameters().size();

		int offset = 0;
		VectorXd parameters(size);
		for(int i = 0; i < (int) m_layer_stack.size(); i++) {
			VectorXd layer_parameters = m_layer_stack[i]->getParameters(); 
			parameters.segment(offset, layer_parameters.size()) = layer_parameters;
			offset += layer_parameters.size();			
		}

		return parameters;
	}

	VectorXd getGradient() 
	{
		int size = 0;
		for(int i = 0; i < (int) m_layer_stack.size(); i++) 
			size += m_layer_stack[i]->getGradient().size();

		int offset = 0;
		VectorXd gradient(size);
		for(int i = 0; i < (int) m_layer_stack.size(); i++) {
			VectorXd layer_gradient = m_layer_stack[i]->getGradient(); 
			gradient.segment(offset, layer_gradient.size()) = layer_gradient;
			offset += layer_gradient.size();			
		}

		return gradient;
	}

	void setParameters(VectorXd& parameters)
	{
		int offset = 0;
		for (int i = 0; i < (int) m_layer_stack.size(); i++) {
			int size = m_layer_stack[i]->getParameters().size();
			VectorXd layer_parameters = parameters.segment(offset, size);
			m_layer_stack[i]->setParameters(layer_parameters);
			offset += size;
		}
	}

	void setGradient(VectorXd& gradient)
	{
		int offset = 0;
		for (int i = 0; i < (int) m_layer_stack.size(); i++) {
			int size = m_layer_stack[i]->getParameters().size();
			VectorXd layer_gradient = gradient.segment(offset, size);
			m_layer_stack[i]->setGradient(layer_gradient);
			offset += size;
		}
	}

	int nbLayers() { return m_layer_stack.size(); }
};
}
