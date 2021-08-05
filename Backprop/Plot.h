////////////////////////////////////////////////////////////////////////////////
// Plot: 
// A tool for displaying charts. Uses on a GNUPlot backend.
////////////////////////////////////////////////////////////////////////////////
#pragma once

#include <stdio.h>
#include <iostream>
#include <vector>
#include <string>

using namespace Eigen;

namespace Backprop
{
class Plot
{
  private :

	FILE * m_pipe_p;

	std::vector <VectorXd*> m_lines;
	std::vector <std::string> m_colors;
	
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
				//"set size ratio -1\n"
				//"set yrange [-1:1]\n"
				//"set palette gray\n"
				//"unset xtics\n"
				//"unset ytics\n"
				"set grid\n"
				"unset key\n";

				//TODO: ensure wxt is the best terminal for what we want
				fprintf(m_pipe_p, "%s", configs);
				fprintf(m_pipe_p, "set term wxt title 'Backprop'\n");
				fprintf(m_pipe_p, "set title \"%s\"\n", m_title.c_str());

				// TODO use th e line with the maximum range				
				fprintf(m_pipe_p, "set xrange [%f:%f] \n", 0.f, (double) m_lines[0]->size()); 
		}
	}	

  public :

	Plot(VectorXd * data, const char * title = "Plot", const char * color = "red") 
	  :  m_title(title)
	{
		m_lines.push_back(data);
		m_colors.push_back(color);

		openGP();
		configPlot();
		render();
	}

	void addLine(VectorXd * line, const char * color)
	{
		m_lines.push_back(line);
		m_colors.push_back(std::string(color));
		render();
	}

	void render()
	{
		if (m_pipe_p) 
		{
			fprintf(m_pipe_p, "plot ");
			for (int l = 0; l < (int) m_lines.size(); l++) 
			{
				fprintf(m_pipe_p, "'-' lc rgb '%s' with lines", m_colors[l].c_str());
				if (l != (int) (m_lines.size() - 1))
					fprintf(m_pipe_p, ", ");
			}
			fprintf(m_pipe_p, "\n");

			for (int l = 0; l < (int) m_lines.size(); l++)
			{
				VectorXd line = (*m_lines[l]);
				
				for(int i = 0; i < line.size(); i++) 
					fprintf(m_pipe_p, "%f \n", line(i));

				fprintf(m_pipe_p, "\ne\n");

				fflush(m_pipe_p);
			}


			////////////////////////////////////////////////


/*
			printf("plot ");
			for (int l = 0; l < (int) m_lines.size(); l++) 
			{
				printf("'-' lc rgb '%s' with lines", m_colors[l].c_str());
				if (l != (int) (m_lines.size() - 1))
					printf( ", ");
			}
			printf("\n");

			for (int l = 0; l < (int) m_lines.size(); l++)
			{
				VectorXd line = (*m_lines[l]);
				
				for(int i = 0; i < line.size(); i++) 
					printf("%f \n", line(i));

				printf("\ne\n");

				fflush(m_pipe_p);
			}
*/

		}


	}

	~Plot()
	{
		pclose(m_pipe_p);
	}
};

}

