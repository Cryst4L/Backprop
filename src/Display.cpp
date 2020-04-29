#include "Display.h"


void Display::openGNUPlot()
{
	m_pipe_p = popen("gnuplot -persist", "w");

	if (!m_pipe_p)
	{
		std::cout << "Failed to open GNUPlot !\n"
		          << "Please ensure it is installed ..."
		          << std::endl;
	}
}

void Display::configPlot()
{
	if (m_pipe_p)
	{
		const char* configs =
			"set size ratio -1\n"
			"set palette gray\n"
			"unset xtics\n"
			"unset ytics\n"
			"unset key\n";

			fprintf(m_pipe_p, "%s", configs);
			fprintf(m_pipe_p, "set term wxt title 'Backprop - Display'\n");
			fprintf(m_pipe_p, "set title \"%s\"\n", m_title.c_str());
	}
}	

Display::Display(Eigen::MatrixXd& frame, const char* title) 
  :  m_frame(frame), m_title(title)
{
	openGNUPlot();
	configPlot();
	render();
}

void Display::render()
{
	if (m_pipe_p) 
	{
			fprintf(m_pipe_p, "plot '-' matrix with image\n");
			fflush(m_pipe_p);

			Eigen::MatrixXd frame = m_frame;
//			Eigen::MatrixXd frame = (*m_frame_p);
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

Display::~Display()
{
	pclose(m_pipe_p);
}


