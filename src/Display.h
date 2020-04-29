#pragma once
#include "Eigen/Core"
#include <stdio.h>
#include <iostream>
#include <vector>

////////////////////////////////////////////////////////////////////////////////
// A tool for displaying matrices based on a GNUPlot backend.
////////////////////////////////////////////////////////////////////////////////

class Display
{
  private :

	FILE* m_pipe_p;

	Eigen::MatrixXd& m_frame;
	
	std::string m_title;

	void openGNUPlot();

	void configPlot();

  public :

	Display(Eigen::MatrixXd& frame, const char* title);

	void render();

	~Display();
};

