////////////////////////////////////////////////////////////////////////////////
// Display: 
// A tool for displaying matrices. Uses on a GNUPlot backend.
////////////////////////////////////////////////////////////////////////////////
#pragma once

#include <stdio.h>
#include <iostream>
#include <vector>

using namespace Eigen;

namespace Backprop
{
class Display
{
  private :

	FILE * m_pipe_p;

	MatrixXd * m_frame_p;
	
	std::string m_title;

	void openGP()
	{
		m_pipe_p = popen("gnuplot -persist", "w");

		if (!m_pipe_p)
		{
			std::cout << "Failed to open GNUPlot !\n"
			          << "Please ensure it is installed ..."
			          << std::endl;
		}
	}

	void configPlot()
	{
		if (m_pipe_p)
		{
			const char* configs =
				"set size ratio -1\n"
//				"set cbrange [-1:1]\n"
				"set palette gray\n"
				"unset xtics\n"
				"unset ytics\n"
				"unset key\n";

				//TODO: ensure wxt is the best terminal for what we want
				fprintf(m_pipe_p, "%s", configs);
				fprintf(m_pipe_p, "set term wxt title 'Backprop'\n");
				fprintf(m_pipe_p, "set title \"%s\"\n", m_title.c_str());
		}
	}	

  public :

	Display(MatrixXd * frame, const char* title) 
	  :  m_frame_p(frame), m_title(title)
	{
		openGP();
		configPlot();
		render();
	}

	void render()
	{
		if (m_pipe_p) 
		{
				fprintf(m_pipe_p, "plot '-' matrix with image\n");
				fflush(m_pipe_p);

				MatrixXd frame = (*m_frame_p);
				for (int i = frame.rows() - 1; i >= 0; i--)
				{
					for(int j = 0; j < frame.cols(); j++)
						fprintf(m_pipe_p, "%f ", frame(i,j));

					fprintf(m_pipe_p, "\n");
					fflush(m_pipe_p);
				}

				fprintf(m_pipe_p, "\ne\n");
				fflush(m_pipe_p);
			}
	}

	~Display()
	{
		pclose(m_pipe_p);
	}
};

}

